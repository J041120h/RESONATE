"""Auxiliary pre-training targets: highly variable reference genes outside the panel (Methods, Training)."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .correlations import cross_corr
from .reftypes import normalized_gene_moments, seurat_dispersion_norm


def build_aux(counts: sp.csr_matrix, symbols, panel_names, panel_cols, REF, emb, *, n_pool=2000, min_mean=0.0125,
              max_mean=3.0, target_sum=1e4):
    """Pool of ``n_pool`` highly variable reference genes that are not panel genes and have an embedding.

    counts (n_ref, n_genes) raw reference counts, symbols their names, panel_cols the reference columns of the panel
    genes, REF (n_ref, G) the panel reference values, emb a ``GeneEmbeddings``. Returns the arrays that the
    pre-training stage reads: sparse counts of the pool, panel library, embeddings V_aux, correlation A_aux (pool x
    panel genes) and cosine S_aux."""
    symbols = np.asarray(symbols).astype(str)
    pos = set(panel_names)
    _, mean, var = normalized_gene_moments(counts)
    dn, lmean, _ = seurat_dispersion_norm(mean, var)
    first = {}
    for i, s in enumerate(symbols):
        first.setdefault(s, i)
    in_mean = (lmean >= min_mean) & (lmean <= max_mean)
    elig = np.array([(s not in pos) and emb.has(s) and np.isfinite(dn[i]) and first[s] == i and in_mean[i]
                     for i, s in enumerate(symbols)])
    if elig.sum() < n_pool:
        n_pool = int(elig.sum())
        if n_pool < 50:
            raise ValueError(f"only {int(elig.sum())} reference genes are eligible as auxiliary targets (not in the "
                             "panel, with an embedding, mean in range); check gene names / embeddings")
    pool = np.argsort(-np.where(elig, dn, -np.inf), kind="stable")[:n_pool]
    pool_sym = [str(symbols[i]) for i in pool]
    lib = np.asarray(counts[:, panel_cols].sum(1)).ravel().astype(np.float64)
    Cpool = counts[:, pool].astype(np.float32).tocsr()
    vc = lib > 0
    Xa = np.log1p(sp.diags(target_sum / np.where(vc, lib, 1.0)) @ Cpool.astype(np.float64)).toarray().astype(np.float32)
    A_aux = cross_corr(Xa[vc], REF[vc])
    V_aux = emb.lookup(pool_sym)
    return dict(pool_symbols=np.array(pool_sym), V_aux=V_aux, A_aux=A_aux, counts=Cpool, panel_library=lib,
                target_sum=float(target_sum))
