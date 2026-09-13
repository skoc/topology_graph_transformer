"""LTA padding and the gamma=0 identity path."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topology import LayerwiseTopologyAggregation, TopologyAwareAttentionBias


def _batch(real_counts=(8, 5), n=8, d_t=4):
    b = len(real_counts)
    s = torch.zeros(b, n, d_t)
    adj = torch.zeros(b, n, n)
    node_mask = torch.zeros(b, n, dtype=torch.bool)
    for i, ni in enumerate(real_counts):
        s[i, :ni] = torch.randn(ni, d_t)
        node_mask[i, :ni] = True
        adj[i, :ni, :ni] = (torch.rand(ni, ni) > 0.6).float()
        adj[i].fill_diagonal_(0.0)
    return s, adj, node_mask


def test_lta_shape_and_padding():
    s, adj, node_mask = _batch()
    lta = LayerwiseTopologyAggregation(d_t=4, hidden_dim=16, n_layers=2)
    s_new = lta.update(s, lta.aggregator(adj), node_mask)
    assert s_new.shape == s.shape
    assert torch.isfinite(s_new).all()
    assert torch.all(s_new[~node_mask] == 0)


def test_lta_gamma_zero_is_identity():
    s, adj, node_mask = _batch()
    lta = LayerwiseTopologyAggregation(d_t=4, hidden_dim=16, n_layers=2)
    assert torch.all(lta.gamma == 0)
    delta = lta.inject(0, lta.update(s, lta.aggregator(adj), node_mask))
    assert torch.all(delta == 0)


def test_attention_bias_padding():
    _, _, node_mask = _batch()
    b, n = node_mask.shape
    c = torch.rand(b, n, n)
    bias = TopologyAwareAttentionBias(num_heads=3, num_bins=8)(c, node_mask)
    assert bias.shape == (b, 3, n, n)
    pad = ~node_mask
    for i in range(b):
        for j in torch.where(pad[i])[0]:
            assert torch.all(bias[i, :, j, :] == 0)
            assert torch.all(bias[i, :, :, j] == 0)


def test_tcgt_forward():
    from model import TCGT

    b, n = 2, 7
    node_mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 0, 0, 0]], dtype=torch.bool)
    x = torch.randn(b, n, 514) * node_mask.unsqueeze(-1)
    adj = torch.zeros(b, n, n)
    adj[0, :5, :5] = 1
    adj[1, :4, :4] = 1
    adj[:, range(n), range(n)] = 0
    merge = torch.rand(b, n, n)
    toposcan = torch.rand(b, n, 5, 4)
    ph = torch.rand(b, 3, 800)

    model = TCGT(num_classes=2, hidden_dim=48, n_layers=2, num_heads=2, ffn_dim=64, use_ph=True, use_bias=True, use_lta=True)
    model.lta.gamma.data.fill_(1.0)
    logits = model(x, node_mask, adj=adj, merge=merge, toposcan=toposcan, ph=ph)
    assert logits.shape == (2, 2)
    logits.sum().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    test_lta_shape_and_padding()
    test_lta_gamma_zero_is_identity()
    test_attention_bias_padding()
    test_tcgt_forward()
    print("ok")
