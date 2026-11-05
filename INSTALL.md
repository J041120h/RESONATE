# Installing RESONATE

RESONATE is **one pure-Python package** (install name `resonate-st`, import name `resonate`, command `resonate`). The
same package runs on a CPU or on an NVIDIA GPU: `resonate.Config(device=...)` chooses the device, and the default is
CUDA when PyTorch can see a GPU. The only thing that differs between a CPU and a GPU installation is the PyTorch build.

---

## pip (CPU)

```bash
python -m venv .venv && source .venv/bin/activate          # Python 3.10 or 3.11
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU-only PyTorch (small download)
pip install "git+https://github.com/J041120h/RESONATE.git"
resonate version
```

From a clone (adds the sample data and the tests):

```bash
git clone https://github.com/J041120h/RESONATE.git && cd RESONATE
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[test]"
pytest -q                     # unit + end-to-end tests on the CPU
pytest -q -m "not slow"       # unit tests only, seconds
```

`pip install -e ".[notebook]"` adds Jupyter and matplotlib for the tutorial notebook.

## conda (CPU)

```bash
conda env create -f environment.yml
conda activate resonate
pip install -e . --no-deps            # add the resonate package itself (from the clone)
```

`environment.yml` pins the versions validated below and takes PyTorch from the CPU wheel index.

## GPU

Install a CUDA build of PyTorch that matches your NVIDIA driver **before** RESONATE, following
<https://pytorch.org/get-started/locally/>, for example

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124     # CUDA 12.4 wheels
pip install "git+https://github.com/J041120h/RESONATE.git"
python -c "import torch; print(torch.cuda.is_available())"               # must print True
```

In `environment.yml`, replace the CPU index line by the CUDA index (see the comment in the file). Nothing else changes.
`pytest -q -m gpu` runs the GPU test (it is skipped when no GPU is visible).

On a GPU the defaults follow the paper's numerics: TF32 matrix multiplies, bf16 inside the attention term and a fused
attention kernel whose backward pass is not deterministic. For tighter run-to-run agreement, at some cost in speed:
`Config(tf32=False, attn_bf16=False, attn_impl="sdpa_det")`. On a CPU pass `tf32=False, attn_bf16=False`.

---

## Validated versions

| package | clean CPU install (`environment.yml`) | Python 3.10 environment (CPU tests; GPU test and equivalence runs) |
|---|---|---|
| Python | 3.11.16 | 3.10.20 |
| torch | 2.15.0+cpu | 2.5.1+cu124 |
| numpy | 1.26.4 | 1.26.4 |
| scipy | 1.17.1 | 1.15.3 |
| pandas | 2.3.3 | 2.3.3 |
| scikit-learn | 1.9.1 | 1.7.2 |
| anndata | 0.12.19 | 0.11.4 |
| scanpy | 1.11.5 | 1.11.5 |
| harmonypy | 2.1.0 | 2.0.0 |
| igraph | 1.0.0 | 1.0.0 |
| leidenalg | 0.12.0 | 0.11.0 |
| pynndescent | 0.6.0 | 0.6.0 |

* Clean CPU install (2026-11-05, Linux x86-64): a fresh clone installed with `pip install .` into a new Python 3.11
  virtual environment (PyTorch from the CPU wheel index, everything else from PyPI); full test suite, sample checksums,
  the README quick start, benchmark and save/reload snippets, and the command line on the sample data.
* Python 3.10 environment: full test suite on the CPU (30 passed, 1 skipped: the GPU test); the GPU test and the
  equivalence runs against the paper model (see the README) on one NVIDIA L40S (driver 580, CUDA 12.4 build of PyTorch).

The reference cell types come from Leiden clustering after Harmony, so other versions of `igraph`, `leidenalg`,
`harmonypy`, `pynndescent` or `scanpy` can give slightly different clusters and therefore slightly different
predictions. `pynndescent` 0.7 does not work with scanpy 1.11 and is excluded by the package requirements; the package
also requires `numpy < 2` and `scanpy < 1.12`.

## Hardware

Measured with the package on Linux (peak memory = maximum resident set size of the process).

| workload | memory | time |
|---|---|---|
| sample data, quick start of the README (CPU, at most 12 epochs per stage) | **16 GB RAM or more recommended** (peak about 11 GiB) | QUICKSTART_TIME |
| sample data, default training on the CPU (up to 500 epochs, early stopping) | same | `fit` about 19 min on 5 CPU cores |
| MERFISH fetal heart, full dataset (250,926 spatial cells, 73,248 reference cells, 236 genes) | `prepare`: peak about 13 GB RAM | `prepare` about 33 min on 8 CPU cores; `fit` 11-14 min on one NVIDIA L40S GPU |

A GPU is not needed for the sample data. For datasets of the size of the paper's (hundreds of thousands of cells), use a
CUDA GPU for `fit`; `prepare` runs its nearest-neighbour searches on the GPU when `device="cuda"`, and Harmony runs on
the CPU.

For panels with thousands of genes the type-specific reference correlations take `T x G x G` floats (`T` reference cell
types, `G` measured + target genes; about 2 GB for 16 types and 5,600 genes): choose a moderate target set at `prepare`
time and predict further genes afterwards with `model.predict(genes, reference=..., gene_embeddings=...)`. For very
large cell numbers, `Config(data_on_host=True)` keeps the cell x gene matrices in host memory.

## Gene embeddings

The sample data ship the embedding rows they need. For your own data, build the full scGPT-spatial token-embedding table
(human gene symbols, 512 dimensions) from the scGPT-spatial V1 checkpoint (figshare doi:10.6084/m9.figshare.28356068,
CC BY 4.0; the folder must contain `best_model.pt` and `vocab.json`):

```python
import resonate
resonate.extract_scgpt_spatial("scGPT_spatial_v1/", "scgpt_spatial_human.npz")   # needs only PyTorch
```

Any other table with one vector per gene symbol works as well (`resonate.load_gene_embeddings`: `.npz`, `.csv`,
DataFrame or AnnData).
