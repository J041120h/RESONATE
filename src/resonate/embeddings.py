"""Gene embeddings: container, loaders and optional foundation-model extraction.

RESONATE needs one fixed vector per gene (scGPT-spatial token embeddings by default). A table is a set of gene symbols
and an (n_genes, dim) matrix; genes missing from the table share one learnable vector in the model.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


class GeneEmbeddings:
    """Gene symbols -> fixed embedding vectors."""

    def __init__(self, genes, matrix, name: str = "custom"):
        genes = [str(g) for g in genes]
        matrix = np.asarray(matrix, np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(genes):
            raise ValueError(f"matrix shape {matrix.shape} does not match {len(genes)} genes")
        if len(set(genes)) != len(genes):
            raise ValueError("duplicate gene symbols in the embedding table")
        if not np.isfinite(matrix).all():
            raise ValueError("embedding matrix contains NaN / inf")
        self.genes, self.matrix, self.name = genes, matrix, name
        self._pos = {g: i for i, g in enumerate(genes)}

    @property
    def dim(self) -> int:
        return self.matrix.shape[1]

    def __len__(self):
        return len(self.genes)

    def has(self, gene: str) -> bool:
        return gene in self._pos

    def lookup(self, genes) -> np.ndarray:
        """Rows for ``genes`` (all must be present)."""
        return self.matrix[[self._pos[g] for g in genes]]

    def panel(self, genes):
        """(V, oov): embedding rows for ``genes`` and a mask of genes without an embedding. Missing rows are filled
        with the mean of the present rows (the model replaces them by one shared learnable vector)."""
        oov = np.array([g not in self._pos for g in genes])
        if oov.all():
            raise ValueError("none of the panel genes has an embedding; check gene symbols (e.g. HGNC symbols)")
        V = np.zeros((len(genes), self.dim), np.float32)
        for i, g in enumerate(genes):
            if not oov[i]:
                V[i] = self.matrix[self._pos[g]]
        V[oov] = V[~oov].mean(0)
        return V, oov

    def save(self, path):
        np.savez_compressed(path, genes=np.array(self.genes), embeddings=self.matrix, name=self.name)

    @classmethod
    def load(cls, path):
        return load_gene_embeddings(path)


def cosine_matrix(V: np.ndarray, oov: np.ndarray) -> np.ndarray:
    """(G, G) cosine similarity of the embeddings; rows / columns of genes without an embedding are zero."""
    G = V.shape[0]
    S = np.zeros((G, G), np.float32)
    inv = ~oov
    if inv.sum() == 0:
        return S
    Vi = V[inv].astype(np.float64)
    Vn = (Vi / (np.linalg.norm(Vi, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
    C = np.clip(Vn @ Vn.T, -1.0, 1.0)
    ii = np.where(inv)[0]
    S[np.ix_(ii, ii)] = C
    return S


def load_gene_embeddings(source, name: str | None = None) -> GeneEmbeddings:
    """Load embeddings from a ``.npz`` (keys ``genes``, ``embeddings``), a ``.csv`` / ``.tsv`` (first column = gene
    symbol, other columns = dimensions; optionally gzipped), a ``pandas.DataFrame`` (index = symbols), a
    ``(genes, matrix)`` tuple, an ``AnnData`` (``var_names`` x ``X``), or an existing ``GeneEmbeddings``."""
    if isinstance(source, GeneEmbeddings):
        return source
    if isinstance(source, tuple) and len(source) == 2:
        return GeneEmbeddings(source[0], source[1], name or "custom")
    try:
        import pandas as pd
        if isinstance(source, pd.DataFrame):
            return GeneEmbeddings(source.index.astype(str), source.to_numpy(np.float32), name or "custom")
    except ImportError:
        pass
    if hasattr(source, "var_names") and hasattr(source, "X"):
        X = source.X.toarray() if hasattr(source.X, "toarray") else np.asarray(source.X)
        return GeneEmbeddings(source.var_names.astype(str), X, name or "custom")
    p = Path(source)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found. Download the shipped table with `resonate download-embeddings` or see "
                                "the documentation (How to choose a reference and embeddings).")
    if p.suffix == ".npz":
        z = np.load(p, allow_pickle=False)
        return GeneEmbeddings(z["genes"], z["embeddings"], name or (str(z["name"]) if "name" in z.files else p.stem))
    import pandas as pd
    sep = "\t" if ".tsv" in p.name else ","
    df = pd.read_csv(p, sep=sep, index_col=0)
    return GeneEmbeddings(df.index.astype(str), df.to_numpy(np.float32), name or p.stem)


def extract_scgpt_spatial(checkpoint_dir, out_path=None, genes=None) -> GeneEmbeddings:
    """Read the token-embedding table of an scGPT-spatial checkpoint (``best_model.pt`` + ``vocab.json``).

    Needs only PyTorch. The checkpoint is a third-party release (see its licence); download it from the scGPT-spatial
    repository. ``genes`` optionally restricts the table (e.g. to the genes of your panel and reference)."""
    import torch
    d = Path(checkpoint_dir)
    sd = torch.load(d / "best_model.pt", map_location="cpu", weights_only=False)
    table = sd["encoder.embedding.weight"].float().numpy()
    vocab = json.load(open(d / "vocab.json"))
    keep = [g for g in vocab if (genes is None or g in set(genes)) and not g.startswith("<")]
    emb = GeneEmbeddings(keep, table[[vocab[g] for g in keep]], "scgpt_spatial")
    if out_path is not None:
        emb.save(out_path)
    return emb


# ---------------------------------------------------------------------------------------------------------------------
# downloadable tables (URL and checksum are filled in once the table is deposited; until then build it with
# extract_scgpt_spatial from the scGPT-spatial checkpoint)
EMBEDDING_REGISTRY = {
    "scgpt_spatial_human": dict(url="https://zenodo.org/records/ZENODO_RECORD_ID/files/scgpt_spatial_human.npz",
                                sha256=None, size_mb=70),
}


def download_embeddings(name: str = "scgpt_spatial_human", dest=None) -> Path:
    """Download a registered embedding table into ``dest`` (default ``~/.cache/resonate``) and return its path."""
    import urllib.request
    info = EMBEDDING_REGISTRY[name]
    if "ZENODO_RECORD_ID" in info["url"]:
        raise RuntimeError("The embedding tables have not been published yet (placeholder URL). Build the table "
                           "yourself with resonate.embeddings.extract_scgpt_spatial(...).")
    dest = Path(dest or os.path.expanduser("~/.cache/resonate"))
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / Path(info["url"]).name
    if not out.exists():
        urllib.request.urlretrieve(info["url"], out)
    if info.get("sha256"):
        h = hashlib.sha256(out.read_bytes()).hexdigest()
        if h != info["sha256"]:
            raise RuntimeError(f"checksum mismatch for {out}")
    return out
