# Equivalence with the paper model

The package is a cleaned re-implementation of the deployed RESONATE v4 code (research folders of the project). This page
reports whether it reproduces the deployed run on one dataset. All numbers below are written by `dev/make_equivalence_page.py`
from the JSON files in `equivalence/` (produced by `dev/equivalence_fetal_heart.py`).

**Dataset and reference run.** MERFISH human fetal heart (Farah et al. 2024): 250,926 cells, 236-gene panel, 73,248 reference
cells; held-out set 0 (48 genes, 47 pass the top-N rule), 25,094 test cells. The deployed run is the paper run
(`merfish_fetal_heart`, draw 0): top-N Pearson **0.4395**, Spearman 0.3550.
Hardware: one NVIDIA L40S, paper numerics (TF32, bf16 attention, fused attention).

## 1. The model code is identical

Deployed weights (`model.pt`) and the deployed cache arrays were fed through the package network.

| inputs | max abs difference to the deployed predictions | top-N Pearson |
|---|---|---|
| deployed cache arrays | 0.0e+00 | 0.439483 |
| package `prepare()` output, Leiden types re-computed | 0.50 (mean 0.0017; correlation 0.99983) | 0.439555 |
| package `prepare()` output, deployed Leiden types injected | 0.01 (mean 0.0000; correlation 1.00000) | 0.439483 |

With the deployed inputs the predictions are bit-for-bit identical, so the network, the readout, the attention term and the
prediction path are the deployed ones. (The deployed checkpoint keys are loaded with `strict=True`.)

## 2. Preparation reproduces the deployed caches

`resonate.prepare` was run on the raw inputs (counts of the fetal-heart panel, the reference, the scGPT-spatial token table) and every
array was compared with the deployed cache. Maximum absolute difference / fraction of equal entries:

| array | Leiden types re-computed | deployed types injected (`type_key`) |
|---|---|---|
| spatial input `X` | 0.0e+00 / 1.0000 | 0.0e+00 / 1.0000 |
| smoothed input `X_input` | 0.0e+00 / 1.0000 | 0.0e+00 / 1.0000 |
| reference lookup (all genes) | 0.0e+00 / 1.0000 | 0.0e+00 / 1.0000 |
| reference lookup (gene folds) | 0.0e+00 / 1.0000 | 0.0e+00 / 1.0000 |
| reference lookup for pre-training | 0.0e+00 / 1.0000 | 0.0e+00 / 1.0000 |
| pooled reference correlation | 0.0e+00 / 1.0000 | 0.0e+00 / 1.0000 |
| type-specific correlations `A_t` | 8.8e-01 / 0.0000 | 0.0e+00 / 1.0000 |
| spatial cell-type votes | 5.9e+01 / 0.9545 | 0.0e+00 / 1.0000 |
| reference cell-type votes | 6.1e+01 / 0.9564 | 0.0e+00 / 1.0000 |
| type distance | 1.8e-02 / 0.0703 | 8.3e-07 / 0.1328 |
| held-out truth (log scale) | 3.3e-07 / 0.5711 | 3.3e-07 / 0.5711 |

* Spatial smoothing, the reference searches, the gene-fold lookups, the correlations, the auxiliary gene pool and the embeddings are
  bit-identical. The Harmony-corrected cell-type votes are identical too once the reference cell types are the same.
* **Reference cell types are the one step that does not reproduce exactly**: Leiden clustering of the reference at the same resolution
  (0.2938) gives the same number of types but a slightly different partition
  (adjusted Rand index 0.953 with the deployed one), as the nearest-neighbour graph and clustering libraries are
  not bit-reproducible across builds and thread counts. Everything that depends on the types (`A_t`, votes, type distances) inherits
  this difference. `prepare(type_key=...)` takes given cell types, which is how the right-hand column was produced.
* Spatial neighbours: 99.77 % of cells have the identical set of 15 neighbours. The deployed search ran on
  float32 coordinates in which each section is shifted by up to 300,000 micrometres, which rounds distances; for the cells that differ
  the package (exact float64 search within each section, as the Methods describe) is the more accurate one. This was verified by reproducing
  the deployed neighbours exactly with a float32 search over all sections.

## 3. Retraining from scratch

`resonate.fit` with default settings (seed as listed) on the package-prepared inputs; same test cells and the same top-N genes.

| inputs | seed | top-N Pearson | top-N Spearman | top-N aligned MSE | correlation of predictions with the deployed ones | stage 1 epochs (best) | stage 2 epochs (best), lambda |
|---|---|---|---|---|---|---|---|
| Leiden re-computed | 0 | 0.4366 | 0.3530 | 0.2593 | 0.9733 | 200 (185) | 18 (3), 0.5 |
| Leiden re-computed | 1 | 0.4285 | 0.3466 | 0.2611 | 0.9613 | 193 (178) | 18 (3), 0.5 |
| Leiden re-computed | 2 | 0.4384 | 0.3584 | 0.2595 | 0.9539 | 210 (195) | 17 (2), 0.75 |
| deployed types injected | 0 | 0.4372 | 0.3537 | 0.2596 | 0.9842 | 263 (248) | 16 (1), 0.5 |
| *deployed paper run* | 0 | 0.4395 | 0.3550 | 0.2584 | 1 | 144 (129) | 16 (1), 0.25 |

**Deployed code, other seeds.** To separate run-to-run noise from a systematic difference, the deployed training code (read-only copy
of the project, same flags) was re-run with other seeds:

| code | seed | top-N Pearson | top-N Spearman | stage 1 epochs (best) | stage 2 epochs (best), lambda |
|---|---|---|---|---|---|
| deployed (paper run) | 0 | 0.4395 | 0.3550 | 144 (129) | 16 (1), 0.25 |
| deployed | 1 | 0.4337 | 0.3504 | 195 (180) | 18 (3), 0.5 |
| deployed | 2 | 0.4382 | 0.3546 | 95 (80) | 17 (2), 0.5 |
| package (Leiden re-computed) | 0, 1, 2 | mean 0.4345 (min 0.4285, max 0.4384) | | | |


Deployed code: mean 0.4371 over 3 runs.

Runs differ in the seed (initialisation, batch order, masks), the length of training varies from run to run (early stopping), and the
GPU kernels used by the paper settings (TF32, bf16 attention, fused attention with an atomic backward) are not bit-deterministic.
The calibrations that depend only on the data reproduce exactly: lookup affine of stage 1
0.9965579481 / -0.0038759276 (deployed 0.9965579481 / -0.0038759276) and of stage 2
0.3125937486 / 0.1368267226 (deployed 0.3125937486 / 0.1368267226).

## Verdict

* Network and prediction code: identical to the deployed model (bit-identical predictions from the deployed weights and inputs).
* Data preparation: identical except for the Leiden partition of the reference and a float32 artefact in 0.23 % of spatial-neighbour sets.
* Retrained scores (top-N Pearson, 3 seeds each): package mean 0.4345 (range 0.4285 to 0.4384); deployed code mean
  0.4371 (range 0.4337 to 0.4395). The package mean is 0.0026 lower. With three seeds per code base this
  difference is smaller than the seed-to-seed range of either (about 0.0099), so it is compatible with noise;
  a systematic shift of a few thousandths cannot be excluded or confirmed from six runs.

Not covered by this test: other datasets, multi-GPU training (not in the package), the cross-omics mode (not in the package).
