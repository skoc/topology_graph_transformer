"""Topology-Conditioned Graph Transformer (TC-GT)."""
import torch
import torch.nn as nn

from local_topo import LocalTopoEncoder
from ph import GraphPHEncoder
from topology import LayerwiseTopologyAggregation, TopologyAwareAttentionBias


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        inner = num_heads * self.head_dim

        self.q = nn.Linear(hidden_dim, inner)
        self.k = nn.Linear(hidden_dim, inner)
        self.v = nn.Linear(hidden_dim, inner)
        self.out = nn.Linear(inner, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, x, attn_bias=None, key_padding_mask=None):
        b, n, _ = x.shape
        shape = (b, n, self.num_heads, self.head_dim)
        q = self.q(x).view(shape).transpose(1, 2) * self.scale
        k = self.k(x).view(shape).transpose(1, 2)
        v = self.v(x).view(shape).transpose(1, 2)

        logits = q @ k.transpose(-1, -2)
        if attn_bias is not None:
            logits = logits + attn_bias
        if key_padding_mask is not None:
            logits = logits.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        attn = self.dropout(logits.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(b, n, -1)
        return self.out(out)


class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim, ffn_dim, num_heads, dropout):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.attn = MultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_bias=None, key_padding_mask=None):
        x = x + self.dropout(self.attn(self.attn_norm(x), attn_bias, key_padding_mask))
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class TCGT(nn.Module):
    """Graph transformer over tissue graphs, conditioned on topology in three ways.

    g_PH conditions the CLS token before the layers and is concatenated to it
    again for classification; the merge-height bias enters every attention
    map; LTA refreshes and reinjects per-node topology states at every layer.
    """

    def __init__(
        self,
        node_dim=514,
        num_classes=2,
        hidden_dim=546,
        n_layers=6,
        num_heads=6,
        ffn_dim=1024,
        dropout=0.1,
        ph_dim=800,
        num_filtrations=3,
        topo_dim=32,
        num_slices=5,
        use_ph=False,
        use_bias=False,
        use_lta=False,
    ):
        super().__init__()
        self.use_ph = use_ph
        self.use_bias = use_bias
        self.use_lta = use_lta
        self.n_layers = n_layers

        in_dim = node_dim + (topo_dim if use_lta else 0)
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        nn.init.trunc_normal_(self.cls, std=0.02)

        self.layers = nn.ModuleList(
            EncoderLayer(hidden_dim, ffn_dim, num_heads, dropout) for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

        if use_lta:
            self.local_encoder = LocalTopoEncoder(num_slices=num_slices, out_dim=topo_dim)
            self.lta = LayerwiseTopologyAggregation(topo_dim, hidden_dim, n_layers)
        if use_bias:
            self.topo_bias = TopologyAwareAttentionBias(num_heads)
        if use_ph:
            self.ph_encoder = GraphPHEncoder(ph_dim, hidden_dim, num_filtrations)

        head_dim = hidden_dim * 2 if use_ph else hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(head_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, node_mask, adj=None, merge=None, toposcan=None, ph=None):
        """x: (B, N, node_dim), node_mask: (B, N) True on real nodes."""
        s = None
        if self.use_lta:
            s = self.local_encoder(toposcan) * node_mask.unsqueeze(-1)
            x = torch.cat([x, s], dim=-1)

        h = self.input_proj(x)

        z_ph = self.ph_encoder(ph) if self.use_ph else None
        cls = self.cls.expand(h.size(0), -1, -1)
        if self.use_ph:
            cls = cls + z_ph.unsqueeze(1)  # early fusion
        h = torch.cat([cls, h], dim=1)

        # CLS is always attendable and never carries a topology state
        padding = torch.cat([torch.zeros_like(node_mask[:, :1]), ~node_mask], dim=1)

        attn_bias = None
        if self.use_bias:
            bias = self.topo_bias(merge, node_mask)
            attn_bias = nn.functional.pad(bias, (1, 0, 1, 0))

        agg = self.lta.aggregator(adj) if self.use_lta else None
        for i, layer in enumerate(self.layers):
            if self.use_lta:
                s = self.lta.update(s, agg, node_mask)
                delta = self.lta.inject(i, s) * node_mask.unsqueeze(-1)
                h = torch.cat([h[:, :1], h[:, 1:] + delta], dim=1)
            h = layer(h, attn_bias, padding)

        out = self.final_norm(h)[:, 0]
        if self.use_ph:
            out = torch.cat([out, z_ph], dim=-1)  # late fusion
        return self.classifier(out)
