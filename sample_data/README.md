# RESONATE tutorial sample data (MERFISH human fetal heart)

Small sample data (about 12 MB) for the quick start in the main `README.md` and the tutorial at
<https://j041120h.github.io/RESONATE_tutorial/>. It runs on a CPU.

| file | content |
|---|---|
| `fetal_heart_spatial.h5ad` | 12,000 cells x 236 genes (raw counts, int16): a square window (about 1.65 mm side) centred on section R78_4C12 (13 PCW heart). `obs`: `section`, `stage`, `author_fine` (authors' cell-type annotation), `split` (train / val / test); `obsm["spatial"]` in micrometres |
| `fetal_heart_reference.h5ad` | 8,011 cells x 2,244 genes (raw counts, int16): stratified subsample (by the authors' `author_label`) of the matched scRNA-seq reference after QC. Genes: the 236 panel genes, the 2,000 highly variable genes used as auxiliary pre-training targets, and marker genes for the unseen-gene demonstration. `obs`: `donor`, `author_label` |
| `scgpt_spatial_embeddings_subset.npz` | scGPT-spatial token embeddings (512 dimensions) of the 2,244 genes above (`genes`, `embeddings`) |
| `held_out_genes.txt` | the 48 panel genes held out in the tutorial's benchmark (held-out set 0 of the paper) |
| `sample_info.json`, `SHA256SUMS` | counts and checksums |
| `zenodo_metadata.json` | draft metadata for a future Zenodo deposit of these files (not yet deposited) |

The spatial window and the reference subsample were cut from the quality-controlled inputs of the paper's MERFISH
fetal-heart dataset.

## Download

The files are part of this repository. Either clone it (`git clone https://github.com/J041120h/RESONATE.git`; the
files are in `RESONATE/sample_data/`) or download single files:

```bash
BASE=https://raw.githubusercontent.com/J041120h/RESONATE/main/sample_data
for f in fetal_heart_spatial.h5ad fetal_heart_reference.h5ad scgpt_spatial_embeddings_subset.npz \
         held_out_genes.txt sample_info.json SHA256SUMS; do
  curl -fsSLO "$BASE/$f"
done
sha256sum -c SHA256SUMS          # every line must end in "OK"
```

## Provenance, licence, citation

* **MERFISH data and scRNA-seq reference** (counts, annotations): Farah, E. N. *et al.* Spatially organized cellular
  communities form the developing human heart. *Nature* 627, 854-864 (2024), doi:10.1038/s41586-024-07171-z. Deposit:
  Dryad doi:10.5061/dryad.w0vt4b8vp, mirrored at Zenodo record 10420367 (title "Integrative single-cell multi-modal
  analyses reveal detailed spatial cellular organization directing human heart morphogenesis"). **Licence: CC0 1.0**
  (Zenodo API metadata `license: cc-zero`, checked 2026-11-04; Dryad deposits are CC0). Redistribution of the subsample is
  permitted; please still cite the paper. Changes: QC, subsampling, restriction to genes, int16 storage.
* **Gene embeddings**: rows of the token-embedding table of scGPT-spatial V1 (Wang, C. *et al.* scGPT-spatial: Continual
  Pretraining of Single-Cell Foundation Model for Spatial Transcriptomics, bioRxiv 2025, doi:10.1101/2025.02.05.636714).
  Weights: figshare doi:10.6084/m9.figshare.28356068 (**CC BY 4.0**, API metadata checked 2026-11-04); code: MIT.
  The rows are unmodified copies; if you use or redistribute `scgpt_spatial_embeddings_subset.npz`, attribute
  scGPT-spatial as required by CC BY 4.0.

The RESONATE code is under the MIT licence (`../LICENSE`); the files in this folder keep the licences above (data CC0 1.0,
embedding rows CC BY 4.0).
