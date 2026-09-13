"""Graph-level PH on a tiny synthetic graph. Skips if GUDHI is missing."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topology import FILTRATIONS, filtration_values, merge_height_matrix, undirected_edges


class TinyGraph:
    def __init__(self, n, edges):
        src = [u for u, v in edges] + [v for u, v in edges]
        dst = [v for u, v in edges] + [u for u, v in edges]
        self._n = n
        self._src = np.array(src, dtype=np.int64)
        self._dst = np.array(dst, dtype=np.int64)

    def num_nodes(self):
        return self._n

    def edges(self):
        return self._src, self._dst


def _triangle():
    return TinyGraph(3, [(0, 1), (1, 2), (2, 0)])


def test_filtration_and_merge():
    g = _triangle()
    values = filtration_values(g, "degree")
    assert values.shape == (3,)
    assert np.all(np.isfinite(values))
    c = merge_height_matrix(values, undirected_edges(g))
    assert c.shape == (3, 3)
    assert np.allclose(c, c.T)


def test_ph_vector_size():
    try:
        import gudhi  # noqa: F401
    except ImportError:
        print("skip test_ph_vector_size (gudhi not installed)")
        return
    from ph import graph_ph_features

    g = _triangle()
    feat = graph_ph_features(g, "degree", resolution=(8, 8))
    assert feat.shape == (2 * 8 * 8,)
    assert np.all(np.isfinite(feat))

    stacked = np.stack([graph_ph_features(g, f, resolution=(8, 8)) for f in FILTRATIONS])
    assert stacked.shape[0] == len(FILTRATIONS)


if __name__ == "__main__":
    test_filtration_and_merge()
    test_ph_vector_size()
    print("ok")
