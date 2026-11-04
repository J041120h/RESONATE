# How the model works

This page summarises the Methods of the paper and states which setting of the package corresponds to which statement. All
defaults of `resonate.Config` are the deployed ones.

## Preparation (`resonate.prepare`)

1. **Normalisation.** Spatial counts: library size over measured genes only, scaled to the median library, `log1p`.
   Reference: scaled to 10,000 over the panel genes, `log1p`.
2. **Smoothing** (MaxFuse): $\tilde x_c = 0.3\,x_c + \frac{0.7}{15}\sum_{c'} x_{c'}$ over the 15 nearest cells in a 30-PC space
   of the standardised measured genes, separately for spatial cells and reference cells (`smooth_weight`, `k_smooth`).
3. **Reference neighbours.** The 100 reference cells with highest cosine similarity in a joint 50-PC space of the smoothed
   profiles (genes standardised within each dataset), no batch correction. Their mean gives the lookup in $M$.
4. **Reference cell types.** Leiden clusters of the reference (2,000 highly variable genes, PCA, Harmony by donor, resolution
   bisected to give 13 to 18 clusters; clusters under 50 cells pooled).
5. **Cell-type mixtures.** The 100 nearest reference cells after Harmony correction with spatial sections and donors as batches:
   $p_c(t) = (n_c(t)+\pi_t)/(K+1)$. The type of a spatial cell is the arg-max.
6. **Gene folds.** Measured genes are split at random into 20 folds; the searches of steps 3 and 5 are repeated using only genes
   outside each fold, and a measured target reads the search that excludes its fold. Unmeasured targets read the all-genes search.
7. **Correlations.** Per-type correlations of reference training cells (90 % of the reference), shrunk toward the pooled
   within-type correlation with $n_0 = 200$; negative partners from the uncentred pooled correlation.
8. **Auxiliary pre-training targets.** 2,000 highly variable reference genes outside the panel with embeddings.

## Model (`resonate.fit`)

* $M = \tfrac12\mu_g + \tfrac12(\beta_1\bar x_{cg}+\beta_0)$; $\mu_g$ = MLP (96-96-96) of the embedding, fixed after a warm-up
  at the start of each stage because other terms could otherwise absorb a uniform shift.
* $R = \sum_t p_c(t)\,[\sum_{P^+_t} w^+(\tilde x-\mu) - \alpha\sum_{P^-} w^-(\tilde x-\mu)]$, up to 128 positive and 8 negative
  partners, softmax weights from one pair-scoring MLP with a rank-one per-type adapter.
* $A$ = cross-attention from the target embedding over transformer-encoded tokens of the measured genes; token features =
  own value, the 15 spatial neighbours' values and the 15 neighbours' reference-type distances.

## Training

Stage 1 (reference): batch 4,096, learning rate 4e-3; stage 2 (spatial): batch 1,024, learning rate 2e-3; Adam, early
stopping with patience 15, at most 500 epochs, seed 0. Each batch hides 20 % of the target genes independently per cell
(loss weight 0.6) and 5 % identically in all cells (weight 0.4). Stage 1 predicts the targets in the reference with 256 of the
2,000 auxiliary genes added per batch. In stage 2, 20 % of the measured genes (detected in at least 10 % of training cells)
are set aside as stand-ins for unmeasured genes: they are inputs but never targets, they select the checkpoint, and they
select $\lambda$ from $\{0,\tfrac14,\tfrac12,\tfrac34,1\}$.

## What the package leaves out

Compared with the research code, the package drops experimental branches that were tested and not adopted (per-type mean
head, gamma term, a sparse partner gather, a posteriori joint weighting of R and A), multi-GPU training (the large cortex
panel used two GPUs; `Config(data_on_host=True)` and a larger GPU are the single-device route), the model-selection tooling
and all cluster paths. Cross-omics (RNA to protein) prediction is future work.
