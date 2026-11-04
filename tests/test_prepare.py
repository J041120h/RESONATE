import numpy as np
import pytest

import resonate
from resonate.toy import make_toy_data


def test_shapes_and_contract(toy_prep):
    p = toy_prep
    G = p.n_genes
    assert G == len(p.vis) + len(p.targets) == p.meta["G"]
    assert p.X.shape == p.X_input.shape == (p.n_cells, G) and p.REF.shape[1] == G
    assert not p.X[:, p.targets].any() and not p.X_input[:, p.targets].any()      # targets are never inputs
    assert p.A_t.shape == (len(p.type_names), G, G)
    assert p.votes_sp.shape == (p.n_cells, 21, len(p.type_names)) and (p.votes_sp.sum(2) == 100).all()
    assert (p.gene_slot[p.targets] == 20).all() and set(np.unique(p.gene_slot[p.vis])) <= set(range(20))
    assert np.isclose(p.pi.sum(), 1.0)
    assert set(p.tr) | set(p.va) | set(p.te) == set(range(p.n_cells))
    assert p.NB.shape == (p.n_cells, 15) and (p.sections[p.NB] == p.sections[:, None]).all()
    assert np.isnan(p.r_ref_cv[:, p.targets]).all() and np.isfinite(p.r_ref).all()
    assert np.allclose(np.diag(p.type_dist), 0) and p.type_dist.max() == pytest.approx(1.0)


def test_no_leakage_from_held_out_counts(toy, toy_cfg, toy_prep):
    """Held-out genes are removed before normalisation: rewriting their counts changes nothing the model reads."""
    sp, ref, emb = toy
    held = list(toy_prep.truth_genes)
    sp2 = sp.copy()
    cols = [sp2.var_names.get_loc(g) for g in held]
    X = sp2.X.copy()
    X[:, cols] = np.random.default_rng(5).integers(0, 50, (X.shape[0], len(cols)))
    sp2.X = X
    p2 = resonate.prepare(sp2, ref, emb, target_genes=held, gene_order=list(toy_prep.genes), config=toy_cfg, verbose=False)
    for k in ("X", "X_input", "r_ref", "votes_sp", "labels_sp", "NB"):
        assert np.array_equal(getattr(p2, k), getattr(toy_prep, k)), k
    assert not np.array_equal(p2.truth_counts, toy_prep.truth_counts)             # only the evaluation truth moved


def test_save_load_roundtrip(toy_prep, tmp_path):
    toy_prep.save(tmp_path / "p")
    q = resonate.Prepared.load(tmp_path / "p")
    for k in ("X", "A_t", "votes_sp", "tr", "genes"):
        assert np.array_equal(getattr(q, k), getattr(toy_prep, k))
    assert q.meta["G"] == toy_prep.meta["G"]


def test_input_validation(toy, toy_cfg):
    sp, ref, emb = toy
    with pytest.raises(ValueError, match="exactly one"):
        resonate.prepare(sp, ref, emb, config=toy_cfg, verbose=False)
    with pytest.raises(ValueError, match="absent from the reference"):
        resonate.prepare(sp, ref[:, ref.var_names[:20]].copy(), emb, heldout_fraction=0.2, config=toy_cfg, verbose=False)
    bad = sp.copy(); bad.X = bad.X * 0.37
    with pytest.raises(ValueError, match="raw non-negative counts"):
        resonate.prepare(bad, ref, emb, heldout_fraction=0.2, config=toy_cfg, verbose=False)
    nocoord = sp.copy(); del nocoord.obsm["spatial"]
    with pytest.raises(ValueError, match="coordinates"):
        resonate.prepare(nocoord, ref, emb, heldout_fraction=0.2, config=toy_cfg, verbose=False)


def test_unmeasured_targets_outside_the_spatial_panel(toy, toy_cfg):
    sp, ref, emb = toy
    p = resonate.prepare(sp, ref, emb, target_genes=["GENE070", "GENE071"], config=toy_cfg, verbose=False)
    assert p.n_genes == sp.n_vars + 2 and p.truth_counts is None and len(p.vis) == sp.n_vars


def test_given_reference_types_replace_leiden(toy, toy_cfg):
    sp, ref, emb = toy
    r = ref.copy()
    r.obs["ct"] = ["T" + t for t in r.obs["cell_type"]]
    p = resonate.prepare(sp, r, emb, heldout_fraction=0.2, type_key="ct", config=toy_cfg, verbose=False)
    assert sorted(p.type_names) == ["T0", "T1", "T2", "T3"] and p.meta["type_source"] == "ct"
    assert np.array_equal(np.array(p.type_names)[p.ref_codes], r.obs["ct"].values)
    from resonate.prep.reftypes import pool_labels
    codes, names = pool_labels(["a"] * 60 + ["b"] * 70 + ["c"] * 3, 50)
    assert names == ["b", "a", "small"] and codes[-1] == 2
