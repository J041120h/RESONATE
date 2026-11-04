# Inputs and outputs

## What you provide

| argument of `prepare` | content | required fields |
|---|---|---|
| `spatial` | `AnnData` of the **measured** genes | raw counts in `.X` (or `layers[spatial_layer]`); unique gene symbols in `var_names`; coordinates in `obsm["spatial"]` (any unit); optional `obs["section"]` (one value per tissue section), optional `obs["split"]` in `{train, val, test}` |
| `reference` | `AnnData` of a matched single-cell reference | raw counts of **all** genes you use (measured panel genes and the target genes); unique symbols; optional `obs["donor"]` (batch for correction) |
| `gene_embeddings` | a table gene symbol to vector | `GeneEmbeddings`, `.npz` (`genes`, `embeddings`), `.csv`, or a DataFrame |
| `target_genes` | the genes to predict | each must be in the reference; targets that are also in `spatial.var_names` are **held out** (see below) |
| `heldout_fraction` | alternative to `target_genes` for benchmarking | fraction of the measured genes to hold out |

All counts must be raw (non-negative integers). The package normalises: spatial profiles are scaled to the median
library size **of the measured genes** and `log1p`-transformed; the reference is scaled to 10,000 counts over the panel
genes (measured + target) and `log1p`-transformed. Genes outside the panel enter only as auxiliary pre-training targets.

Held-out genes are removed from the model input **before** normalisation. Their measured values are kept only in
`Prepared.truth_counts` for scoring and are never read during training.

Without `obs["split"]`, cells are split 80/10/10 at random (stratified by `label_key` when given). Training cells fit the
weights, validation cells select the checkpoint, test cells are for scoring only; at prediction time all cells are used
for smoothing, reference searches and spatial neighbours.

## What `prepare` returns

A `Prepared` object (`prep.save(dir)` / `Prepared.load(dir)`). Main fields, with `G` = measured genes followed by
target genes, `T` reference cell types:

| field | shape | meaning |
|---|---|---|
| `genes`, `vis`, `targets` | `G`, measured idx, target idx | gene axis of the model |
| `X`, `X_input` | cells x `G` | log-normalised values; smoothed values. Target columns are 0 |
| `NB` | cells x 15 | spatial neighbours (within a section) |
| `REF` | reference cells x `G` | log-normalised reference over the panel |
| `ref_codes`, `type_names`, `type_dist` | n_ref, `T`, `T x T` | reference Leiden cell types and the distance between types |
| `A_t` | `T x G x G` | type-specific reference correlations (shrunk toward the pooled within-type correlation) |
| `votes_sp`, `votes_ref` | cells x 21 x `T` | cell-type mixtures under each of 20 gene folds and the all-genes slot |
| `r_ref`, `r_ref_cv`, `nbr` | cells x `G`, cells x `G`, cells x 100 | reference lookup (all genes / gene-fold) and the 100 nearest reference cells |
| `tr`, `va`, `te` | indices | split |
| `truth_counts`, `truth_genes`, `lib_full` | | measured values of held-out genes, for `score_heldout` |

## What `predict` returns

An `AnnData` with cells (in the order of `spatial`) by the requested genes.

* `X`: predicted expression on the model's log scale, i.e. `log1p` of counts normalised to the median library size of
  the measured genes. This is **not** a count estimate; compare it with `log1p(normalize_total(...))` of data of the same
  kind. Do not compare absolute levels across genes without checking calibration; correlation across cells is the
  quantity the model optimises.
* `obs["split"]`, `obsm["spatial"]`, `var["is_fit_target"]`, `uns["resonate"]["lambda_"]` (the attention weight chosen).

`model.save(dir)` stores the weights (`model.pt`), the configuration and training summary (`model.json`).
`Resonate.load(dir, prep)` restores a model for the same prepared inputs.

## Scoring

`resonate.score_heldout(prep, pred)` returns Pearson, Spearman and aligned MSE across test cells for (a) all held-out
genes and (b) the top-N genes: held-out genes detected in at least 10 % of cells, limited to the 100 with the highest
mean count (the paper's headline number). Genes with a constant truth or prediction score 0 and stay in the mean.
