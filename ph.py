"""Graph-level topological descriptor g_PH.

Persistent homology over the degree / betweenness / HKS superlevel
filtrations, vectorised as persistence images and fused across filtrations.
"""
import numpy as np
import torch
import torch.nn as nn

from topology import FILTRATIONS, filtration_values, undirected_edges

try:
    import gudhi
except ImportError:  # keeps import-time failures away from training runs
    gudhi = None


def persistence_diagrams(graph, filtration, tau=1.0, max_dim=1):
    """H0/H1 diagrams of the superlevel filtration on a tissue graph.

    Superlevel filtration on u is computed as a sublevel filtration on -u;
    an edge enters with max(-u_i, -u_j) so it appears once both endpoints do.
    persistence_dim_max=True is required for 1-complexes, otherwise GUDHI
    stops at H0 and never reports cycles.
    """
    if gudhi is None:
        raise ImportError("GUDHI is required for persistent homology: pip install gudhi")

    values = -filtration_values(graph, filtration, tau=tau)
    st = gudhi.SimplexTree()
    for v, fv in enumerate(values):
        st.insert([v], filtration=float(fv))
    for u, v in undirected_edges(graph):
        st.insert([int(u), int(v)], filtration=float(max(values[u], values[v])))

    st.make_filtration_non_decreasing()
    st.compute_persistence(min_persistence=0.0, persistence_dim_max=True)

    diagrams = []
    for dim in range(max_dim + 1):
        pairs = np.array(st.persistence_intervals_in_dimension(dim), dtype=np.float64)
        if len(pairs):
            finite = pairs[np.isfinite(pairs)]
            pairs[~np.isfinite(pairs)] = float(finite.max()) if len(finite) else 1.0
        else:
            pairs = np.empty((0, 2))
        diagrams.append(pairs)
    return diagrams


def persistence_image(diagram, resolution=(20, 20), bandwidth=1.0):
    """Persistence-weighted Gaussian kernel density over (birth, persistence)."""
    if len(diagram) == 0:
        return np.zeros(resolution, dtype=np.float32)

    births = diagram[:, 0]
    pers = diagram[:, 1] - births

    b_lo, b_hi = births.min(), births.max()
    if b_hi == b_lo:
        b_lo, b_hi = b_lo - 0.1, b_hi + 0.1
    p_hi = pers.max() if pers.max() > 0 else 1.0

    x = np.linspace(b_lo, b_hi, resolution[0])
    y = np.linspace(0.0, p_hi, resolution[1])
    X, Y = np.meshgrid(x, y)

    image = np.zeros_like(X)
    for b, p in zip(births, pers):
        image += p * np.exp(-((X - b) ** 2 + (Y - p) ** 2) / (2 * bandwidth ** 2))
    return image.astype(np.float32)


def graph_ph_features(graph, filtration, tau=1.0, resolution=(20, 20), bandwidth=1.0):
    """Flattened H0 and H1 persistence images for one filtration."""
    diagrams = persistence_diagrams(graph, filtration, tau=tau)
    images = [persistence_image(d, resolution, bandwidth).ravel() for d in diagrams]
    return np.concatenate(images)


class GraphPHEncoder(nn.Module):
    """Per-filtration projection, then self-attention fusion + mean pool.

    Input  (B, F, ph_dim) persistence-image vectors
    Output (B, d_model)   the unified descriptor g_PH
    """

    def __init__(self, ph_dim, d_model, num_filtrations=len(FILTRATIONS), nhead=2, dropout=0.1):
        super().__init__()
        self.project = nn.ModuleList(
            nn.Sequential(nn.Linear(ph_dim, d_model), nn.LayerNorm(d_model))
            for _ in range(num_filtrations)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2 * d_model,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, ph):
        z = torch.stack([p(ph[:, i]) for i, p in enumerate(self.project)], dim=1)
        return self.fusion(z).mean(dim=1)
