import numpy as np
import torch

from resonate.train import fit_global_affine, make_train_masks, tier_loss


def test_masks_disjoint_targets_never_inputs():
    G, pool, targets = 50, np.arange(0, 40), np.arange(40, 50)
    rng = np.random.default_rng(0)
    inp, A, B = make_train_masks(64, G, pool, targets, rng, 0.20, 0.05)
    assert not (A & B).any() and not (inp & A).any() and not (inp & B).any()
    assert not inp[:, targets].any() and not A[:, targets].any() and not B[:, targets].any()
    assert ((A.sum(1) <= 8) & (A.sum(1) >= 6)).all()   # ceil(20 % of 40) drawn, minus overlap with the batch mask
    assert (B.sum(1) == 2).all() and (B == B[0]).all()  # ceil(5 %), identical in every cell of the batch
    assert (A.any(0) != A.all(0)).any()                # the per-cell mask differs between cells


def test_tier_loss_weights():
    pred = torch.zeros(2, 3)
    tgt = torch.ones(2, 3)
    A = torch.tensor([[1, 0, 0], [0, 0, 0]], dtype=torch.bool)
    B = torch.tensor([[0, 1, 0], [0, 0, 0]], dtype=torch.bool)
    assert float(tier_loss(pred, tgt, A, B, True, True)) == 0.6 + 0.4


def test_affine_recovers_line():
    rng = np.random.default_rng(1)
    r = rng.normal(size=(30, 10))
    X = 2.0 * r - 0.5
    a, b = fit_global_affine(r, X, np.arange(30), np.arange(10))
    assert abs(a - 2.0) < 1e-6 and abs(b + 0.5) < 1e-6
