"""Gene-gene correlations in the reference: pooled, within-type and shrunk type-specific (Methods, Correlation term)."""
from __future__ import annotations

import numpy as np

from .neighbors import zscore_cols


def corr_matrix(M):
    """Pearson correlation of the columns of M (z-scored with the +1e-8 stabiliser), as (m, m) float32."""
    Rz = zscore_cols(M)
    return ((Rz.T @ Rz) / Rz.shape[0]).astype(np.float32)


def cross_corr(Xa, Xb):
    """Correlation between columns of Xa (n, ka) and Xb (n, kb)."""
    return ((zscore_cols(Xa).T @ zscore_cols(Xb)) / Xa.shape[0]).astype(np.float32)


def reference_split(n_ref, seed=0, val_frac=0.10):
    """Reference cells -> (train, validation): rng(seed + 42) permutation, first ceil(10 %) validation."""
    perm = np.random.default_rng(seed + 42).permutation(n_ref)
    n_val = max(1, int(np.ceil(val_frac * n_ref)))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def type_correlations(REF, codes, names, train_idx, *, min_type_cells=50, shrink_n0=200):
    """A_t (T, G, G): per-type correlation shrunk toward the pooled within-type correlation,
    ``(n_t R_t + n0 A_W) / (n_t + n0)``; types with fewer than ``min_type_cells`` training cells (and 'small') use A_W.
    Also returns A_ref_tr (G, G), the correlation over all training cells without centring by type (negative
    partners), and A_W."""
    T = len(names)
    REFt, ct = REF[train_idx], np.asarray(codes)[train_idx].astype(np.int64)
    n_t = np.bincount(ct, minlength=T)
    A_ref_tr = corr_matrix(REFt)
    E = REFt.copy()
    for t in range(T):
        idx = np.where(ct == t)[0]
        if len(idx):
            E[idx] -= E[idx].mean(0, keepdims=True)
    A_W = corr_matrix(E)
    del E
    G = REF.shape[1]
    A_t = np.empty((T, G, G), np.float32)
    own = []
    for t in range(T):
        if names[t] == "small" or n_t[t] < min_type_cells:
            A_t[t] = A_W
            own.append(False)
            continue
        R = corr_matrix(REFt[ct == t])
        A_t[t] = (n_t[t] * R + shrink_n0 * A_W) / (n_t[t] + shrink_n0)
        own.append(True)
    return dict(A_t=A_t, A_W=A_W, A_ref_tr=A_ref_tr, n_t=n_t, own=own)
