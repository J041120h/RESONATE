# Changelog

## 0.2.0 (2026-11-05)

* `resonate.impute(spatial, reference, embeddings, genes=..., heldout=...)`: prepare, fit and predict in one call.
* `resonate.score(pred, spatial)`: per-gene Pearson / Spearman of held-out genes as a table (same numbers as
  `score_heldout`).
* `resonate.datasets.fetal_heart_demo()`: downloads, checks and loads the demo data.
* `resonate.pl`: `cell_types`, `gene_scores`, `spatial_compare` (measured vs rank-matched predicted maps),
  `spatial_genes`, `celltype_heatmap`.
* `resonate run` now passes `--heldout-seed` to `prepare` (it always used seed 0 before).
* The existing API (`prepare`, `fit`, `Resonate.predict`, `score_heldout`, the command line) is unchanged.

## 0.1.0 (2026-11-05)

First public release.

* The RESONATE RNA model (mean, correlation and attention terms; reference pre-training and spatial fine-tuning) with
  `prepare` / `fit` / `predict`, a command line, gene-embedding loaders, scoring of held-out genes, tests and tutorial
  sample data (MERFISH human fetal heart).
* Equivalence check against the model used in the paper (`equivalence/`).
* Not included: the cross-omics (RNA to protein) mode, multi-GPU training.
