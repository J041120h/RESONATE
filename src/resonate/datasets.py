"""Demo data: download, check and load the fetal-heart MERFISH sample of the code repository (``sample_data/``)."""
from __future__ import annotations

import hashlib
from collections import namedtuple
from pathlib import Path

DEMO_URL = "https://raw.githubusercontent.com/J041120h/RESONATE/main/sample_data"

# broad groups of the authors' cell types (obs["author_fine"]), used only to read plots; RESONATE does not use them
FETAL_HEART_GROUPS = {
    "vCM compact": ["vCM-LV-Compact", "vCM-RV-Compact"],
    "vCM trabecular": ["vCM-LV-Trabecular", "vCM-RV-Trabecular"],
    "vCM His-Purkinje": ["vCM-His-Purkinje"],
    "vCM AV": ["vCM-LV-AV", "vCM-RV-AV"],
    "vCM proliferating": ["vCM-Proliferating"],
    "Fibroblast / valve": ["VIC", "vFibro", "aFibro", "EPDC"],
    "Endothelial": ["BEC", "VEC", "vEndocardial", "aEndocardial", "LEC"],
    "Pericyte / SMC": ["Pericyte", "VSMC"],
}

Demo = namedtuple("Demo", ["spatial", "reference", "embeddings", "held_out"])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _download(url: str, out: Path):
    import urllib.request
    tmp = out.with_name(out.name + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        f.write(r.read())
    tmp.replace(out)


def fetal_heart_demo(path="data", *, base_url: str = DEMO_URL, verbose: bool = True) -> Demo:
    """Download (if needed), check and load the fetal-heart demo data (about 12 MB).

    The files are written to ``path``; files already there are kept when their SHA-256 checksum matches
    ``SHA256SUMS``. A file that does not match after downloading raises ``RuntimeError``.

    Returns ``Demo(spatial, reference, embeddings, held_out)``:

    * ``spatial``: MERFISH raw counts, 12,000 cells x 236 genes, ``obsm["spatial"]`` in micrometres, ``obs["split"]``,
      the authors' cell types ``obs["author_fine"]`` and eight broad groups of them in ``obs["cell_group"]`` (for plots);
    * ``reference``: scRNA-seq raw counts, 8,011 cells x 2,244 genes, ``obs["donor"]``, ``obs["author_label"]``;
    * ``embeddings``: the scGPT-spatial embeddings of the reference genes (a :class:`~resonate.GeneEmbeddings`);
    * ``held_out``: the 48 panel genes used as held-out genes in the tutorial.

    Data: Farah et al., Nature 2024 (CC0 1.0); embedding rows: scGPT-spatial, Wang et al., bioRxiv 2025 (CC BY 4.0).
    """
    import anndata as ad
    import pandas as pd

    from .embeddings import load_gene_embeddings

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    sums = path / "SHA256SUMS"
    if not sums.exists():
        _download(f"{base_url}/SHA256SUMS", sums)
    expected = dict(reversed(line.split()) for line in sums.read_text().splitlines() if line.strip())
    for name, digest in expected.items():
        out = path / name
        state = "found"
        if not out.exists() or _sha256(out) != digest:
            _download(f"{base_url}/{name}", out)
            state = "downloaded"
            if _sha256(out) != digest:
                raise RuntimeError(f"checksum mismatch for {out}: delete it and try again")
        if verbose:
            print(f"{out}: OK ({state})")

    spatial = ad.read_h5ad(path / "fetal_heart_spatial.h5ad")
    to_group = {label: group for group, labels in FETAL_HEART_GROUPS.items() for label in labels}
    group = spatial.obs["author_fine"].astype(str).map(to_group).fillna("Other")
    spatial.obs["cell_group"] = pd.Categorical(group, categories=list(FETAL_HEART_GROUPS) + ["Other"])
    reference = ad.read_h5ad(path / "fetal_heart_reference.h5ad")
    embeddings = load_gene_embeddings(path / "scgpt_spatial_embeddings_subset.npz")
    held_out = (path / "held_out_genes.txt").read_text().split()
    return Demo(spatial, reference, embeddings, held_out)
