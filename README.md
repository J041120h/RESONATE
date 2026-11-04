# RESONATE

Predict genes that were never measured in a spatial transcriptomics panel, using a matched single-cell reference,
foundation-model gene embeddings and spatial context.

For a spatial cell *c* and an unmeasured gene *g*, RESONATE predicts

    x_hat(c, g) = M(c, g) + R(c, g) + lambda * A(c, g)

a **mean term** (the gene's embedding-predicted level and its mean among similar reference cells), a **correlation
term** (adjustment from co-measured genes correlated with *g* in the reference, mixed over reference cell types) and an
**attention term** (a transformer over all measured genes of the cell and of its 15 spatial neighbours). Training has two
stages: reference pre-training, then spatial fine-tuning.

> Status: pre-release (`0.1.0.dev0`). Package name on PyPI will be `resonate-st` (`resonate` is taken); the import name
> is `resonate`. See `RELEASE_CHECKLIST.md`.

## Install

```bash
pip install -e .            # from a clone; needs Python >= 3.10 and PyTorch
pip install -e ".[test]"    # + pytest
```

## Use

```python
import anndata as ad, resonate

spatial   = ad.read_h5ad("spatial.h5ad")      # raw counts of the measured genes, obsm["spatial"]
reference = ad.read_h5ad("reference.h5ad")    # raw counts of a matched single-cell reference (measures the targets)
emb       = resonate.load_gene_embeddings("scgpt_spatial_human.npz")

prep  = resonate.prepare(spatial, reference, emb, target_genes=["NPPA", "HCN1", "BMP10"])
model = resonate.fit(prep)                    # reference pre-training + spatial fine-tuning
pred  = model.predict(["NPPA", "HCN1"])       # AnnData, cells x genes (log-normalised)
model.predict(["TBX3"], reference=reference, gene_embeddings=emb)   # any other reference gene, no refit
```

Benchmark mode (hold out measured genes and score them): `prepare(..., heldout_fraction=0.2)` then
`resonate.score_heldout(prep, pred)`.

Command line: `resonate prepare|fit|predict|run ...` (see `resonate --help`).

## Documentation, tests, sample data

* Docs: `mkdocs serve` (sources in `docs/`, built site in `site/`); tutorial notebook on the sample data in `docs/tutorial.ipynb`.
* Tests: `pytest -q` (about a minute on a CPU; `-m "not slow"` skips the end-to-end tests).
* Sample data (MERFISH fetal heart subsample, CC0): `sample_data/`, prepared for Zenodo; see `sample_data/README.md`.
* Equivalence with the deployed paper model: `equivalence/` and `docs/equivalence.md`.

## Licence and citation

Licence: **to be chosen by the authors** (`LICENSE` is a placeholder). Citation: see `CITATION.cff` (placeholder).
