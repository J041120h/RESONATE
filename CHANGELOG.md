# Changelog

## 0.1.0 (2026-11-05)

First public release.

* The RESONATE RNA model (mean, correlation and attention terms; reference pre-training and spatial fine-tuning) with
  `prepare` / `fit` / `predict`, a command line, gene-embedding loaders, scoring of held-out genes, tests and tutorial
  sample data (MERFISH human fetal heart).
* Equivalence check against the model used in the paper (`equivalence/`).
* Not included: the cross-omics (RNA to protein) mode, multi-GPU training.
