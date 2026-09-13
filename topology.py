"""Graph filtrations and the two topology-conditioned learning mechanisms.

Filtration metrics (degree, betweenness, HKS) are shared by the graph-level
descriptor in ph.py and the node-level descriptor in local_topo.py.
"""
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import linalg as spla

FILTRATIONS = ("degree", "betweenness", "hks")


# --------------------------------------------------------------------------
# filtration values on the tissue graph
# --------------------------------------------------------------------------

def _edge_arrays(graph):
    src, dst = graph.edges()
    if hasattr(src, "cpu"):
        src, dst = src.cpu().numpy(), dst.cpu().numpy()
    return np.asarray(src), np.asarray(dst)


def to_networkx(graph):
    """DGL graph -> undirected networkx graph."""
    src, dst = _edge_arrays(graph)
    g = nx.Graph()
    g.add_nodes_from(range(graph.num_nodes()))
    g.add_edges_from(zip(src, dst))
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def heat_kernel_signature(g, tau=1.0, k_eig=50):
    """HKS(i) = sum_k exp(-tau * lambda_k) phi_k(i)^2, min-max scaled to [0, 1]."""
    n = g.number_of_nodes()
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    if n == 1:
        return np.ones(1, dtype=np.float32)

    L = nx.normalized_laplacian_matrix(g)
    k = min(k_eig, n - 1)
    try:
        vals, vecs = spla.eigsh(L, k=k, which="SM", tol=1e-6)
    except Exception:
        vals, vecs = np.linalg.eigh(L.toarray())

    order = np.argsort(vals)
    vals, vecs = vals[order], vecs[:, order]

    # zero eigenvalues carry no diffusion information (one per component)
    keep = vals > 1e-10
    if not keep.any():
        return np.full(n, 0.5, dtype=np.float32)

    hks = (vecs[:, keep] ** 2).dot(np.exp(-tau * vals[keep]))
    if hks.max() > hks.min():
        hks = (hks - hks.min()) / (hks.max() - hks.min())
    else:
        hks = np.full_like(hks, 0.5)
    return hks.astype(np.float32)


def filtration_values(graph, filtration, tau=1.0):
    """Per-node filtration values in [0, 1] for one of FILTRATIONS."""
    g = to_networkx(graph)
    n = g.number_of_nodes()
    if n == 0:
        return np.zeros(0, dtype=np.float32)

    if filtration == "degree":
        v = np.array([d for _, d in g.degree()], dtype=np.float32)
        return v / v.max() if v.max() > 0 else v
    if filtration == "betweenness":
        bc = nx.betweenness_centrality(g, normalized=True)
        return np.array([bc[i] for i in range(n)], dtype=np.float32)
    if filtration == "hks":
        return heat_kernel_signature(g, tau=tau)
    raise ValueError(f"unknown filtration {filtration!r}, expected one of {FILTRATIONS}")


def undirected_edges(graph):
    """Unique (u, v) pairs with u < v; DGL stores both directions."""
    src, dst = _edge_arrays(graph)
    keep = src != dst
    src, dst = src[keep], dst[keep]
    pairs = np.stack([np.minimum(src, dst), np.maximum(src, dst)], axis=1)
    return np.unique(pairs, axis=0) if len(pairs) else pairs.reshape(0, 2)


# --------------------------------------------------------------------------
# pairwise topology metric c_ij (Eq. 2)
# --------------------------------------------------------------------------

def merge_height_matrix(values, edges):
    """c_ij = highest superlevel threshold at which i and j are connected.

    With node values u and edge weight min(u_i, u_j) this is the max-min
    (bottleneck) path value, so a Kruskal pass over edges sorted by
    descending weight fills every cross-component pair as it merges.
    Pairs in different components of the full graph stay 0.
    """
    n = len(values)
    c = np.zeros((n, n), dtype=np.float32)
    if n <= 1 or len(edges) == 0:
        return c

    w = np.minimum(values[edges[:, 0]], values[edges[:, 1]])
    order = np.argsort(-w, kind="stable")
    edges, w = edges[order], w[order]

    parent = list(range(n))
    members = [[i] for i in range(n)]

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (u, v), weight in zip(edges, w):
        ru, rv = find(int(u)), find(int(v))
        if ru == rv:
            continue
        a, b = members[ru], members[rv]
        c[np.ix_(a, b)] = weight
        c[np.ix_(b, a)] = weight
        if len(a) < len(b):
            ru, rv, a, b = rv, ru, b, a
        parent[rv] = ru
        a.extend(b)
        members[rv] = []
    return c


class TopologyAwareAttentionBias(nn.Module):
    """b_topo^(h)(c_ij): bins the merge heights and maps each bin per head.

    The embedding starts at zero so the bias is a no-op at step 0.
    """

    def __init__(self, num_heads, num_bins=32):
        super().__init__()
        self.num_bins = num_bins
        self.bin_embed = nn.Embedding(num_bins + 1, num_heads, padding_idx=0)  # bin 0 = padding
        nn.init.zeros_(self.bin_embed.weight)

    def forward(self, c, node_mask):
        """c: (B, N, N) in [0, 1], node_mask: (B, N) -> bias (B, H, N, N)."""
        valid = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        bins = (c.clamp(0, 1) * (self.num_bins - 1)).long() + 1
        bins = bins * valid.long()
        bias = self.bin_embed(bins) * valid.unsqueeze(-1)
        return bias.permute(0, 3, 1, 2).contiguous()


# --------------------------------------------------------------------------
# layerwise topology aggregation (Eq. 4-6)
# --------------------------------------------------------------------------

class LayerwiseTopologyAggregation(nn.Module):
    """Recurrent per-node topology state, reinjected at every layer.

    s_i^(l) = MLP_S([s_i^(l-1) || mean_{j in N(i)} s_j^(l-1)])
    h_i     <- h_i + gamma_l * W_S s_i^(l)

    gamma is learnable per layer and starts at 0, so the branch does not
    perturb the backbone at the beginning of training.
    """

    def __init__(self, d_t, hidden_dim, n_layers):
        super().__init__()
        self.mlp_s = nn.Sequential(
            nn.Linear(2 * d_t, d_t),
            nn.GELU(),
            nn.Linear(d_t, d_t),
        )
        self.w_s = nn.Linear(d_t, hidden_dim)
        self.gamma = nn.Parameter(torch.zeros(n_layers))

    @staticmethod
    def aggregator(adj):
        """Row-normalised adjacency; clamp keeps isolated nodes at zero."""
        return adj / adj.sum(-1, keepdim=True).clamp(min=1.0)

    def update(self, s, agg, node_mask):
        s_new = self.mlp_s(torch.cat([s, agg @ s], dim=-1))
        return s_new * node_mask.unsqueeze(-1).to(s_new.dtype)

    def inject(self, layer_idx, s):
        return self.gamma[layer_idx] * self.w_s(s)
