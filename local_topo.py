"""Node-level local topological descriptor t_i.

Each k-hop induced subgraph is cut into overlapping slices along its HKS
values; every slice contributes a 4-D token (beta0, beta1, |V|, |E|) and the
slice sequence is encoded into t_i.  Sliding-filtration representation
following Uddin et al. (2026).
"""
import numpy as np
import torch
import torch.nn as nn

from topology import filtration_values, to_networkx

TOKEN_DIM = 4


def _adjacency(n, edges):
    adj = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def _khop_nodes(adj, source, k):
    """Node indices within k hops of source, sorted."""
    seen = {source}
    frontier = [source]
    for _ in range(k):
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    nxt.append(v)
        frontier = nxt
        if not frontier:
            break
    return np.array(sorted(seen), dtype=np.int64)


def _components(members, adj):
    """Number of connected components among members (induced subgraph)."""
    member_set = set(members)
    seen = set()
    count = 0
    for start in members:
        if start in seen:
            continue
        count += 1
        stack = [start]
        seen.add(start)
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v in member_set and v not in seen:
                    seen.add(v)
                    stack.append(v)
    return count


def _slice_token(members, adj):
    """(beta0, beta1, |V|, |E|) of the induced subgraph on members."""
    nv = len(members)
    if nv == 0:
        return np.zeros(TOKEN_DIM, dtype=np.float32)

    member_set = set(members)
    ne = sum(1 for u in members for v in adj[u] if v in member_set) // 2
    beta0 = _components(members, adj)
    beta1 = ne - nv + beta0  # cycle rank of a 1-complex
    return np.array([beta0, max(beta1, 0), nv, ne], dtype=np.float32)


def _slice_bounds(num_slices, window, stride):
    """Overlapping [lo, hi] windows in quantile-rank units of 1 / num_slices."""
    unit = 1.0 / num_slices
    bounds = []
    for n in range(num_slices):
        lo = min(n * stride * unit, 1.0)
        hi = min(lo + window * unit, 1.0)
        bounds.append((lo, hi))
    return bounds


def local_toposcan_sequence(graph, k=2, num_slices=5, window=2, stride=1, tau=1.0):
    """Per-node slice sequences for one tissue graph, shape (N, num_slices, 4)."""
    g = to_networkx(graph)
    n = g.number_of_nodes()
    edges = [(int(u), int(v)) for u, v in g.edges()]
    adj = _adjacency(n, edges)
    hks = filtration_values(graph, "hks", tau=tau)
    bounds = _slice_bounds(num_slices, window, stride)

    seq = np.zeros((n, num_slices, TOKEN_DIM), dtype=np.float32)
    for i in range(n):
        local = _khop_nodes(adj, i, k)
        values = hks[local]
        lo_v, hi_v = float(values.min()), float(values.max())
        span = hi_v - lo_v
        for s, (lo, hi) in enumerate(bounds):
            if span <= 0:
                members = local.tolist()  # constant HKS: the whole neighbourhood
            else:
                thr_lo, thr_hi = lo_v + lo * span, lo_v + hi * span
                members = local[(values >= thr_lo) & (values <= thr_hi)].tolist()
            seq[i, s] = _slice_token(members, adj)
    return seq


class LocalTopoEncoder(nn.Module):
    """Slice sequence -> t_i, with a transformer along the slice axis.

    Mean pooling over slices keeps t_i independent of the slice count.
    """

    def __init__(self, num_slices=5, d_model=32, nhead=2, num_layers=2, out_dim=32, dropout=0.1):
        super().__init__()
        self.token_proj = nn.Linear(TOKEN_DIM, d_model)
        self.pos = nn.Parameter(torch.zeros(1, num_slices, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(nn.Linear(d_model, out_dim), nn.LayerNorm(out_dim))

    def forward(self, seq):
        """seq: (B, N, T, 4) -> (B, N, out_dim)."""
        b, n, t, _ = seq.shape
        z = self.token_proj(seq.reshape(b * n, t, TOKEN_DIM)) + self.pos
        z = self.encoder(z).mean(dim=1)
        return self.head(z).view(b, n, -1)
