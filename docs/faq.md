# FAQ

**Which genes can I predict?** Any gene that the reference measures and that has an embedding. Genes listed in
`target_genes` are modelled during training. Other reference genes can be predicted afterwards without refitting
(`model.predict(genes, reference=..., gene_embeddings=...)`); the paper's biological analyses use this route.

**How many target genes at fit time?** The type-specific correlation tensor is `T x G x G` with `G` = measured + target
genes (for 16 types and 5,600 genes that is about 2 GB). Choose a moderate target set at `prepare` time (a few hundred to a few
thousand) and add genes at prediction time.

**What is a "held-out" gene?** A gene that is measured in your spatial data but removed from the model input to test the
model. Use `heldout_fraction=0.2` or list measured genes in `target_genes`; then `score_heldout` compares prediction with the
measured values. It is the only way to estimate accuracy on genes the model has not seen.

**The predictions are not counts.** They are on the log-normalised scale of the spatial data (median measured-gene library,
`log1p`). Convert with `expm1` for a normalised-count scale; absolute levels of single genes should be treated cautiously.

**It is slow.** `prepare` repeats nearest-neighbour searches and Harmony 21 times (20 gene folds plus all genes); a GPU helps
the searches (`device="cuda"`), Harmony runs on the CPU. `Prepared.save` lets you reuse the result for several fits.
Training time scales with the number of cells and `G x G`.

**GPU out of memory.** Use `Config(data_on_host=True)` (matrices stay in CPU RAM), lower `stage1_batch_size`, or reduce the target set.

**Results differ slightly between runs.** The default numerics follow the paper: TF32 matrix multiplies and bf16 inside the
attention term on CUDA, and a fused attention kernel with non-deterministic backward. Set `Config(tf32=False, attn_bf16=False,
attn_impl="sdpa_det")` for tighter reproducibility at some cost in speed. See the [equivalence page](equivalence.md) for run-to-run spread.

**My tissue is mouse.** Provide a mouse-symbol embedding table. The scGPT-spatial table is human.

**Can I use a reference without donors?** Yes; cell types and mixtures are then not batch-corrected. Provide `batch_key` if
the reference pools samples.

**Can I use several sections?** Yes; put the section name in `obs["section"]`. Neighbours are searched within a section and each
section is its own batch in the cell-type search.

**How do I cite it?** See `CITATION.cff`.

**RNA to protein?** Planned: the same code with a CITE-seq reference. It is not in this release.
