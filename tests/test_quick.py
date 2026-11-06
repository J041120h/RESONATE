"""The one-call helpers (impute, score), the demo-data loader and the plots."""
import hashlib
from pathlib import Path

import matplotlib
import numpy as np
import pytest

import resonate

matplotlib.use("Agg")
SAMPLE = Path(__file__).resolve().parents[1] / "sample_data"


@pytest.mark.slow
def test_impute_score_and_plots(toy, toy_cfg):
    sp, ref, emb = toy
    heldout = list(sp.var_names[:8])
    pred, model = resonate.impute(sp, ref, emb, genes=["GENE100", "GENE150"], heldout=heldout, config=toy_cfg,
                                  device="cpu", return_model=True, verbose=False)
    assert pred.shape == (sp.n_obs, 10) and np.isfinite(pred.X).all()
    assert list(pred.var_names[pred.var["heldout"]]) == heldout
    table = resonate.score(pred, sp)
    assert list(table.columns) == ["pearson", "spearman", "detected", "top_n"]
    assert sorted(table.index) == sorted(heldout) and table["pearson"].is_monotonic_decreasing
    # same numbers as score_heldout on the prepared inputs of the same model
    ref_scores = resonate.score_heldout(model.prepared, pred)
    assert np.isclose(table.loc[table.top_n, "pearson"].mean(), ref_scores["top_n"]["pearson"])
    assert np.isclose(table["pearson"].mean(), ref_scores["all_heldout"]["pearson"])
    sp = sp.copy()
    sp.obs["group"] = sp.obs["cell_type"].astype("category")
    figs = [resonate.pl.cell_types(sp, "group", show=False),
            resonate.pl.gene_scores(table, show=False),
            resonate.pl.spatial_compare(pred, sp, list(table.index[:2]), table=table, show=False),
            resonate.pl.spatial_genes(pred, ["GENE100", "GENE150"], show=False),
            resonate.pl.celltype_heatmap(pred, sp, "group", ["GENE100", "GENE150"], reference=ref,
                                         reference_groupby="cell_type", show=False)]
    assert all(isinstance(f, matplotlib.figure.Figure) for f in figs)


def test_impute_checks_genes(toy):
    sp, ref, emb = toy
    with pytest.raises(ValueError, match="genes to predict"):
        resonate.impute(sp, ref, emb)
    with pytest.raises(ValueError, match="measured in the panel"):
        resonate.impute(sp, ref, emb, genes=[sp.var_names[0]])
    with pytest.raises(ValueError, match="not measured in the panel"):
        resonate.impute(sp, ref, emb, heldout=["GENE150"])


def test_fetal_heart_demo_from_local_copy(tmp_path):
    demo = resonate.datasets.fetal_heart_demo(tmp_path / "data", base_url=SAMPLE.as_uri(), verbose=False)
    assert demo.spatial.shape == (12000, 236) and demo.reference.shape == (8011, 2244)
    assert len(demo.held_out) == 48 and len(demo.embeddings) == 2244
    assert "Other" in demo.spatial.obs["cell_group"].cat.categories
    # a second call reuses the files; a corrupted file is downloaded again
    (tmp_path / "data" / "held_out_genes.txt").write_text("broken")
    demo = resonate.datasets.fetal_heart_demo(tmp_path / "data", base_url=SAMPLE.as_uri(), verbose=False)
    assert len(demo.held_out) == 48
    digest = hashlib.sha256((tmp_path / "data" / "held_out_genes.txt").read_bytes()).hexdigest()
    assert digest in (SAMPLE / "SHA256SUMS").read_text()
