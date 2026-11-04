# Installation

RESONATE needs Python 3.10 or newer and PyTorch (CPU is enough for the tutorial; a CUDA GPU speeds up large datasets).

```bash
git clone https://github.com/J041120h/resonate && cd resonate
python -m venv .venv && source .venv/bin/activate
pip install torch                 # or follow https://pytorch.org for a CUDA build
pip install -e .                  # core
pip install -e ".[test]"          # + pytest
pip install -e ".[docs,notebook]" # + to build this website and run the tutorial
```

Once released: `pip install resonate-st` (the PyPI name `resonate` belongs to another project; the import name is
`resonate`).

Check the installation:

```bash
resonate version
pytest -q -m "not slow"           # seconds
pytest -q                         # about a minute on a laptop CPU
```

## Hardware

| dataset | CPU RAM | GPU |
|---|---|---|
| tutorial sample (12k spatial cells, 8k reference cells, 236 genes) | 8 GB | not needed |
| 250k spatial cells, 73k reference cells, 236 genes (fetal heart) | 32 GB for `prepare` | one 16 GB GPU, about 10 minutes of training |
| panels with thousands of genes | the type-specific correlations take `T x G x G` floats; set `Config(data_on_host=True)` for very large cell counts | 40 GB class GPU |

## Gene embeddings

RESONATE needs an embedding table (gene symbol to vector). See
[Choosing a reference and embeddings](choosing-reference-and-embeddings.md). The tutorial ships the rows it needs;
the full scGPT-spatial table is built with `resonate.extract_scgpt_spatial(checkpoint_dir)` from the authors' checkpoint
or downloaded with `resonate download-embeddings` once published.

## Tested versions

The equivalence test and the tutorial were run with Python 3.10, torch 2.5.1, numpy 1.26.4, scipy 1.15.3, pandas 2.3.3,
scikit-learn 1.7.2, anndata 0.11.4, scanpy 1.11.5, harmonypy 2.0.0, igraph 1.0.0, leidenalg 0.11.0, pynndescent 0.6.0.
The reference cell types come from Leiden clustering and Harmony, so other versions of those libraries can give slightly
different clusters (and therefore slightly different predictions). `pynndescent` 0.7 does not work with scanpy 1.11 on large
reference sets and is excluded by the package requirements.
