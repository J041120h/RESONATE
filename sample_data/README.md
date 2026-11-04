# RESONATE tutorial sample data (MERFISH human fetal heart)

Prepared for a Zenodo deposit (not uploaded). About 12 MB. Used by `docs/tutorial.ipynb`.

| file | content |
|---|---|
| `fetal_heart_spatial.h5ad` | 12,000 cells x 236 genes (raw counts, int16): a square window (about 1.65 mm side) centred on section R78_4C12 (13 PCW heart). `obs`: `section`, `stage`, `author_fine` (authors' cell-type annotation), `split` (train / val / test); `obsm["spatial"]` in micrometres |
| `fetal_heart_reference.h5ad` | 8,011 cells x 2,244 genes (raw counts, int16): stratified subsample (by the authors' `author_label`) of the matched scRNA-seq reference after QC. Genes: the 236 panel genes, the 2,000 highly variable genes used as auxiliary pre-training targets, and marker genes for the unseen-gene demonstration. `obs`: `donor`, `author_label` |
| `scgpt_spatial_embeddings_subset.npz` | scGPT-spatial token embeddings (512 dimensions) of the 2,244 genes above (`genes`, `embeddings`) |
| `held_out_genes.txt` | the 48 panel genes held out in the tutorial's benchmark (held-out set 0 of the paper) |
| `sample_info.json`, `SHA256SUMS` | counts and checksums |

Built by `dev/make_sample_data.py` from the QC'd paper inputs of the MERFISH fetal-heart panel.

## Provenance, licence, citation

* **MERFISH data and scRNA-seq reference** (counts, annotations): Farah, E. N. *et al.* Spatially organized cellular
  communities form the developing human heart. *Nature* 627, 854-864 (2024), doi:10.1038/s41586-024-07171-z. Deposit:
  Dryad doi:10.5061/dryad.w0vt4b8vp, mirrored at Zenodo record 10420367 (title "Integrative single-cell multi-modal
  analyses reveal detailed spatial cellular organization directing human heart morphogenesis"). **Licence: CC0 1.0**
  (Zenodo API metadata `license: cc-zero`, checked 2026-11-04; Dryad deposits are CC0). Redistribution of the subsample is
  permitted; please still cite the paper. Changes: QC, subsampling, restriction to genes, int16 storage.
* **Gene embeddings**: rows of the token-embedding table of scGPT-spatial V1 (Wang, C. *et al.* scGPT-spatial: Continual
  Pretraining of Single-Cell Foundation Model for Spatial Transcriptomics, bioRxiv 2025, doi:10.1101/2025.02.05.636714). Weights: figshare
  doi:10.6084/m9.figshare.28356068 (**CC BY 4.0**, API metadata checked 2026-11-04); code: MIT. Attribution is required;
  the rows are unmodified copies.

The Zenodo record that contains both parts is proposed under **CC BY 4.0** (CC0 data may be bundled under CC BY); see
`zenodo_metadata.json`.
