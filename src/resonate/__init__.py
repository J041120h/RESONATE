"""RESONATE: predict genes never measured in a spatial transcriptomics panel.

Typical use::

    import resonate
    prep  = resonate.prepare(spatial_adata, reference_adata, "scgpt_spatial_human.npz", target_genes=[...])
    model = resonate.fit(prep)
    pred  = model.predict(["NPPA", "HCN1"])        # AnnData, cells x genes
"""
from importlib.metadata import PackageNotFoundError, version

from .api import Resonate, fit, predict
from .config import Config
from .embeddings import GeneEmbeddings, download_embeddings, extract_scgpt_spatial, load_gene_embeddings
from .metrics import score_heldout
from .prep.pipeline import Prepared, prepare

try:
    __version__ = version("resonate-st")
except PackageNotFoundError:          # running from a source tree
    __version__ = "0.1.0.dev0"

__all__ = ["prepare", "fit", "predict", "Resonate", "Prepared", "Config", "GeneEmbeddings", "load_gene_embeddings",
           "extract_scgpt_spatial", "download_embeddings", "score_heldout", "__version__"]
