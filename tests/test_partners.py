import numpy as np

from resonate.model.model import negative_sets, positive_sets


def _toy(T=2, G=12, seed=0):
    rng = np.random.default_rng(seed)
    A_t = rng.normal(size=(T, G, G)).astype(np.float32)
    A_ref = rng.normal(size=(G, G)).astype(np.float32)
    S = rng.normal(size=(G, G)).astype(np.float32)
    return A_t, A_ref, S


def test_partner_rules():
    A_t, A_ref, S = _toy()
    vis = np.arange(0, 9)
    N_idx, N_ok = negative_sets(A_ref, vis, 3)
    P_idx, P_ok = positive_sets(A_t, S, vis, N_idx, N_ok, 4)
    for g in range(12):
        neg = N_idx[g][N_ok[g]]
        assert (A_ref[g, neg] < 0).all() and g not in neg and set(neg) <= set(vis)
        assert np.all(np.diff(A_ref[g, neg]) >= 0)                       # most negative first
        for t in range(2):
            pos = P_idx[t, g][P_ok[t, g]]
            assert len(pos) <= 4 and (A_t[t, g, pos] > 0).all() and g not in pos
            assert not set(pos) & set(neg) and set(pos) <= set(vis)
            assert np.all(np.diff(A_t[t, g, pos]) <= 0)                  # strongest first


def test_rowwise_with_external_genes():
    A_t, A_ref, S = _toy()
    vis = np.arange(0, 9)
    N_idx, N_ok = negative_sets(A_ref[:3], vis, 3, self_idx=-np.ones(3, int))
    P_idx, P_ok = positive_sets(A_t[:, :3], S[:3], vis, N_idx, N_ok, 4, self_idx=-np.ones(3, int))
    N2, ok2 = negative_sets(A_ref, vis, 3)
    P2, pok2 = positive_sets(A_t, S, vis, N2, ok2, 4)
    for g in (9, 10, 11):                 # genes that are not measured: their own column is never a candidate anyway
        n3, o3 = negative_sets(A_ref[[g]], vis, 3, self_idx=[-1])
        assert np.array_equal(n3[0], N2[g]) and np.array_equal(o3[0], ok2[g])
