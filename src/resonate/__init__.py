"""RESONATE: predict genes never measured in a spatial transcriptomics panel.

Typical use::

    import resonate
    pred = resonate.impute(spatial_adata, reference_adata, "scgpt_spatial_human.npz", genes=["NPPA", "HCN1"])

or step by step::

    prep  = resonate.prepare(spatial_adata, reference_adata, "scgpt_spatial_human.npz", target_genes=[...])
    model = resonate.fit(prep)
    pred  = model.predict(["NPPA", "HCN1"])        # AnnData, cells x genes

``resonate.datasets`` loads the demo data, ``resonate.score`` scores held-out genes and ``resonate.pl`` plots.
"""
from importlib.metadata import PackageNotFoundError, version

from . import datasets, pl
from .api import Resonate, fit, predict
from .config import Config
from .embeddings import GeneEmbeddings, download_embeddings, extract_scgpt_spatial, load_gene_embeddings
from .metrics import score_heldout
from .prep.pipeline import Prepared, prepare
from .quick import impute, score

try:
    __version__ = version("resonate-st")
except PackageNotFoundError:          # running from a source tree
    __version__ = "0.2.0"

__all__ = ["impute", "score", "datasets", "pl", "prepare", "fit", "predict", "Resonate", "Prepared", "Config",
           "GeneEmbeddings", "load_gene_embeddings", "extract_scgpt_spatial", "download_embeddings", "score_heldout",
           "__version__"]
