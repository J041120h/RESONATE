import numpy as np

from resonate.metrics import per_gene_pearson, score


def test_perfect_and_degenerate():
    rng = np.random.default_rng(0)
    t = rng.normal(size=(100, 4))
    p = t.copy()
    p[:, 3] = 1.0                                            # constant prediction -> scores 0 but stays in the mean
    s = score(t, p)
    assert abs(s["pearson"] - 0.75) < 1e-9 and s["n_degenerate"] == 1
    assert abs(s["spearman"] - 0.75) < 1e-9


def test_pearson_matches_numpy():
    rng = np.random.default_rng(1)
    a, b = rng.normal(size=(50, 3)), rng.normal(size=(50, 3))
    r = per_gene_pearson(a, b)
    assert np.allclose(r, [np.corrcoef(a[:, j], b[:, j])[0, 1] for j in range(3)])


def test_aligned_mse_identity():
    rng = np.random.default_rng(2)
    t = rng.normal(size=(200, 3))
    p = 3 * t + rng.normal(size=(200, 3))
    s = score(t, p)
    r = np.array(s["per_gene"]["pearson"])
    assert np.allclose(s["per_gene"]["mse"], t.var(0) * (1 - np.clip(r, 0, None) ** 2), atol=1e-8)
