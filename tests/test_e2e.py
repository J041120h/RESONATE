import json
import subprocess
import sys

import numpy as np
import pytest

import resonate
from resonate.metrics import score_heldout

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def fitted(toy_prep, toy_cfg):
    return resonate.fit(toy_prep, toy_cfg, verbose=False)


def test_fit_predict_score(fitted, toy_prep):
    pred = fitted.predict()
    assert pred.shape == (toy_prep.n_cells, len(toy_prep.targets)) and np.isfinite(pred.X).all()
    assert list(pred.var_names) == fitted.target_genes and "spatial" in pred.obsm
    s = score_heldout(toy_prep, pred)
    assert s["top_n"]["pearson"] > 0.0                       # the toy has real signal; a trained model beats chance
    assert 0.0 in fitted.cfg.lambda_grid and fitted.net.a_scale in fitted.cfg.lambda_grid


def test_save_load_same_predictions(fitted, toy_prep, tmp_path):
    fitted.save(tmp_path / "m")
    m2 = resonate.Resonate.load(tmp_path / "m", toy_prep, device="cpu")
    a, b = fitted.predict(as_anndata=False)[0], m2.predict(as_anndata=False)[0]
    assert np.allclose(a, b, atol=1e-5)


def test_extra_gene_path_matches_panel_path(fitted, toy):
    """A target gene predicted through the unseen-gene path (new gene row, built from the reference) must equal its
    prediction through the fitted panel."""
    sp, ref, emb = toy
    genes = fitted.target_genes[:3]
    a = fitted.predict(genes, as_anndata=False)[0]
    b = fitted._predict_extra(genes, np.arange(fitted.prepared.n_cells), ref, emb, 512, 128)
    assert np.allclose(a, b, atol=1e-4), np.abs(a - b).max()


def test_predict_outside_fit_targets(fitted, toy):
    sp, ref, emb = toy
    out = fitted.predict(["GENE100", "GENE150"], reference=ref, gene_embeddings=emb)
    assert out.shape == (fitted.prepared.n_cells, 2) and np.isfinite(out.X).all()
    with pytest.raises(ValueError, match="reference="):
        fitted.predict(["GENE100"])
    with pytest.raises(ValueError, match="not in the reference"):
        fitted.predict(["NOT_A_GENE"], reference=ref, gene_embeddings=emb)


def test_cli_roundtrip(toy, tmp_path):
    sp, ref, emb = toy
    sp.write_h5ad(tmp_path / "sp.h5ad"); ref.write_h5ad(tmp_path / "ref.h5ad"); emb.save(tmp_path / "emb.npz")
    json.dump(dict(device="cpu", n_types_range=[3, 6], n_types_target=4, min_type_cells=20, shrink_n0=50, n_aux_genes=100,
                   aux_per_batch=32, max_epochs=2, patience=2, stage1_batch_size=256, stage2_batch_size=256, tf32=False,
                   attn_bf16=False), open(tmp_path / "cfg.json", "w"))
    run = lambda *a: subprocess.run([sys.executable, "-m", "resonate.cli", *a], check=True, capture_output=True, text=True)
    run("prepare", "--spatial", str(tmp_path / "sp.h5ad"), "--reference", str(tmp_path / "ref.h5ad"), "--embeddings",
        str(tmp_path / "emb.npz"), "--heldout-fraction", "0.2", "--config", str(tmp_path / "cfg.json"), "--out", str(tmp_path / "prep"))
    run("fit", "--prepared", str(tmp_path / "prep"), "--out", str(tmp_path / "model"), "--config", str(tmp_path / "cfg.json"))
    run("predict", "--prepared", str(tmp_path / "prep"), "--model", str(tmp_path / "model"), "--out", str(tmp_path / "pred.h5ad"),
        "--device", "cpu")
    import anndata as ad
    out = ad.read_h5ad(tmp_path / "pred.h5ad")
    assert out.n_obs == sp.n_obs and out.n_vars > 0


def test_measured_genes_cannot_be_predicted(fitted, toy):
    sp, ref, emb = toy
    g = [x for x in sp.var_names if x not in fitted.target_genes][0]
    with pytest.raises(ValueError, match="measured"):
        fitted.predict([g], reference=ref, gene_embeddings=emb)
