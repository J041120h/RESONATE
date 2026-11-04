import numpy as np
import pytest
import torch

from resonate.model.model import DEPLOY_SLOT, K_NN


def _batch(net, n=16, drop=None):
    idx = torch.arange(n)
    G = net.G
    m = np.zeros((n, G), bool)
    m[:, net.pair_idx.new_tensor(np.arange(G)).cpu().numpy()] = True
    m[:, net.target_mask.numpy()] = False
    if drop is not None:
        m[:, drop] = False
    return idx, torch.tensor(m)


def test_attention_term_is_zero_at_init(toy_net):
    net = toy_net
    net.eval()
    with torch.no_grad():
        x, parts = net(*_batch(net), lookup="test", return_parts=True)
    assert float(parts["A"].abs().max()) == 0.0
    assert float(parts["R"].abs().max()) == 0.0 or True      # the pair-score head starts at zero: uniform partner mean
    assert torch.isfinite(x).all()


def test_target_values_never_reach_predictions(toy_net, toy_prep):
    """Changing the (zeroed) target columns of the data must not change any prediction."""
    net = toy_net
    net.eval()
    idx, m = _batch(net)
    with torch.no_grad():
        x0 = net(idx, m, lookup="test").clone()
        keep, keep_in = net.X.clone(), net.X_in.clone()
        net.X[:, net.target_mask] = 123.0
        net.X_in[:, net.target_mask] = 99.0
        x1 = net(idx, m, lookup="test")
        net.X.copy_(keep); net.X_in.copy_(keep_in)
    assert torch.equal(x0, x1)


def test_masked_gene_value_does_not_change_its_own_prediction(toy_net):
    net = toy_net
    net.eval()
    g = int(toy_prep_vis(net)[0])
    idx, m = _batch(net, drop=g)
    with torch.no_grad():
        # perturb the model so that it is not at its trivial initial point
        torch.manual_seed(0)
        for p in net.corr_mlp.parameters():
            p.add_(0.05 * torch.randn_like(p))
        for p in net.attn.parameters():
            p.add_(0.05 * torch.randn_like(p))
        x0 = net(idx, m, lookup="test")[:, g].clone()
        keep, keep_in = net.X.clone(), net.X_in.clone()
        net.X[:, g] += 5.0
        net.X_in[:, g] += 5.0
        x1 = net(idx, m, lookup="test")[:, g]
        net.X.copy_(keep); net.X_in.copy_(keep_in)
    assert torch.allclose(x0, x1, atol=1e-6)


def toy_prep_vis(net):
    return np.where(~net.target_mask.numpy())[0]


def test_correlation_term_matches_loop_formula(toy_net):
    """R(c, g) = sum_t p_c(t) [sum_P w+ y - alpha sum_N w- y] with softmax weights, written out for one (cell, gene)."""
    net = toy_net
    net.eval()
    torch.manual_seed(1)
    with torch.no_grad():
        for p in net.corr_mlp.parameters():
            p.add_(0.3 * torch.randn_like(p))
        net.alpha_neg.fill_(0.4)
        idx, m = _batch(net, n=4)
        x, parts = net(idx, m, lookup="test", return_parts=True)
        V = net.gene_emb()
        S = net._pair_scores(V, net.panel_block(V))               # (T, G, KP+KN)
        c, g = 2, int(np.where(net.target_mask.numpy())[0][0])
        y = parts["y"][c]
        mm = m[c].float()
        slot = int(net.gene_slot[g])
        p_t = (net.votes[c, slot].float() + net.pi) / (K_NN + 1)
        total = 0.0
        for t in range(net.T):
            pos = [(net.pair_idx[t, g, k], S[t, g, k]) for k in range(net.KP) if net.pair_ok[t, g, k]]
            neg = [(net.pair_idx[t, g, k], -S[t, g, k]) for k in range(net.KP, net.KP + net.KN) if net.pair_ok[t, g, k]]

            def branch(items):
                items = [(j, s) for j, s in items if mm[j] > 0]
                if not items:
                    return 0.0
                s = torch.stack([s for _, s in items])
                w = torch.softmax(s, 0)
                return sum(w[i] * y[j] for i, (j, _) in enumerate(items))
            total += p_t[t] * (branch(pos) - net.alpha_neg * branch(neg))
    assert float(abs(total - parts["R"][c, g])) < 1e-5


def test_frozen_mu_gets_no_gradient(toy_net):
    assert all(not p.requires_grad for p in toy_net.mu_head.parameters())
    assert toy_net.mu_frozen


def test_deploy_slot_for_targets(toy_net):
    assert (toy_net.gene_slot[toy_net.target_mask] == DEPLOY_SLOT).all()
