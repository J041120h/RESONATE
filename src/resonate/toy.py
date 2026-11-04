"""A tiny simulated dataset for tests and for trying the API in seconds (not a biological benchmark)."""
from __future__ import annotations

import numpy as np


def make_toy_data(n_spatial=1200, n_ref=1500, n_panel=60, n_other=200, n_types=4, n_sections=2, n_donors=2, seed=0):
    """Return ``(spatial, reference, embeddings)``.

    Cells belong to ``n_types`` types with type-specific mean programs; counts are Poisson. The spatial panel is the first
    ``n_panel`` genes; the reference measures all ``n_panel + n_other`` genes. Gene embeddings are noisy copies of each
    gene's program loadings, so they are informative about co-expression. Spatial cells of one type are spatially
    clustered."""
    import anndata as ad
    import pandas as pd
    from .embeddings import GeneEmbeddings
    rng = np.random.default_rng(seed)
    G = n_panel + n_other
    genes = [f"GENE{i:03d}" for i in range(G)]
    base = rng.gamma(1.0, 0.4, G) + 0.05
    load = np.exp(rng.normal(0, 0.9, (n_types, G)))
    load[rng.random((n_types, G)) < 0.5] = 0.3
    prog = base[None, :] * load

    def cells(n, t):
        lam = prog[t] * rng.gamma(8.0, 1 / 8.0, (n, 1))
        return rng.poisson(lam).astype(np.float32)

    t_sp = rng.integers(0, n_types, n_spatial)
    t_ref = rng.integers(0, n_types, n_ref)
    Xs = cells(n_spatial, t_sp)[:, :n_panel]
    Xr = cells(n_ref, t_ref)
    sec = np.array([f"S{i % n_sections}" for i in range(n_spatial)])
    xy = np.zeros((n_spatial, 2))
    for i in range(n_spatial):
        xy[i] = rng.normal(0, 1, 2) + 6 * t_sp[i] + 100 * (i % n_sections)
    split = np.array(["train"] * n_spatial, object)
    perm = rng.permutation(n_spatial)
    split[perm[:n_spatial // 10]] = "test"
    split[perm[n_spatial // 10:n_spatial // 5]] = "val"
    spatial = ad.AnnData(X=Xs, obs=pd.DataFrame({"section": sec, "split": split, "cell_type": t_sp.astype(str)},
                                                  index=[f"cell{i}" for i in range(n_spatial)]),
                         var=pd.DataFrame(index=genes[:n_panel]))
    spatial.obsm["spatial"] = xy
    reference = ad.AnnData(X=Xr, obs=pd.DataFrame({"donor": [f"D{i % n_donors}" for i in range(n_ref)],
                                                   "cell_type": t_ref.astype(str)},
                                                  index=[f"ref{i}" for i in range(n_ref)]),
                           var=pd.DataFrame(index=genes))
    E = np.log(load.T) @ rng.normal(0, 1, (n_types, 32)) + rng.normal(0, 0.3, (G, 32))
    E = np.hstack([E, rng.normal(0, 0.1, (G, 32))]).astype(np.float32)
    return spatial, reference, GeneEmbeddings(genes, E, "toy")
