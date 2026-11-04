"""Reference cell types: Leiden clusters of donor-corrected reference cells, and the type-to-type distance."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .neighbors import harmony


def seurat_dispersion_norm(mean_raw, var_raw, n_bins=20):
    """Seurat-flavour normalised dispersion (scanpy ``highly_variable_genes``), on the normalised-count scale.
    Returns (normalised dispersion, log1p mean, log dispersion)."""
    import pandas as pd
    mean = np.array(mean_raw, dtype=np.float64, copy=True)
    mean[mean == 0] = 1e-12
    disp = np.array(var_raw, dtype=np.float64) / mean
    disp[disp == 0] = np.nan
    disp = np.log(disp)
    mean = np.log1p(mean)
    df = pd.DataFrame(dict(means=mean, dispersions=disp))
    df["mean_bin"] = pd.cut(df["means"], bins=n_bins)
    g = df.groupby("mean_bin", observed=False)["dispersions"]
    m_b, s_b = g.mean(), g.std(ddof=1)
    one = s_b.isnull()
    s_b[one.values] = m_b[one.values].values
    m_b[one.values] = 0
    dn = (df["dispersions"].values - m_b[df["mean_bin"].values].values) / s_b[df["mean_bin"].values].values
    return dn, df["means"].values, df["dispersions"].values


def normalized_gene_moments(C: sp.csr_matrix, target=1e4):
    """Per-cell scales (target / total counts; 1 for empty cells) and per-gene mean / variance (ddof 1) of the
    library-size-normalised counts."""
    tot = np.asarray(C.sum(1)).ravel().astype(np.float64)
    scales = np.where(tot > 0, target / np.where(tot > 0, tot, 1.0), 1.0)
    Y = sp.diags(scales) @ C.astype(np.float64)
    n = C.shape[0]
    mean = np.asarray(Y.sum(0)).ravel() / n
    sq = np.asarray(Y.multiply(Y).sum(0)).ravel() / n
    var = (sq - mean ** 2) * (n / (n - 1))
    return scales, mean, var


def centroid_dist(E, codes, T):
    C = np.stack([E[codes == t].mean(0) for t in range(T)])
    return np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2).sum(-1))


def pool_labels(labels, min_type_cells=50):
    """Integer codes from arbitrary labels: types ordered by size (largest first); types smaller than
    ``min_type_cells`` are pooled into a last type called ``small``."""
    labels = np.asarray(labels).astype(str)
    u, inv = np.unique(labels, return_inverse=True)
    n = np.bincount(inv)
    big = [i for i in np.argsort(-n, kind="stable") if n[i] >= min_type_cells]
    code_of = {int(i): k for k, i in enumerate(big)}
    names = [str(u[i]) for i in big]
    if len(big) < len(u):
        for i in range(len(u)):
            code_of.setdefault(i, len(big))
        names.append("small")
    return np.array([code_of[int(i)] for i in inv], np.int16), names


def reference_types(counts: sp.csr_matrix, batch=None, *, labels=None, n_hvg=2000, n_pcs=50, n_neighbors=15,
                    t_range=(13, 18), t_target=15, min_type_cells=50, seed=0, log=print):
    """Leiden clusters of the reference cells.

    Recipe: library-size normalise to 1e4, log1p, 2,000 Seurat-dispersion highly variable genes, scale (clip 10), PCA
    (50), Harmony correction by ``batch`` (donor; skipped when ``None``), 15-NN graph, Leiden (igraph, 2 iterations).
    The resolution is bisected so that the number of clusters falls in ``t_range`` (the value closest to ``t_target``
    wins); clusters with fewer than ``min_type_cells`` cells are pooled into one type called ``small``.

    ``labels`` (one per reference cell) replaces the Leiden clustering by given cell types (pooled by ``pool_labels``);
    the embedding and the type distances are computed in the same way.

    Returns dict(codes (n,) int16, names, n_t, resolution, Z (n, n_pcs) the corrected embedding, D (T, T) centroid
    distances divided by their maximum, D_raw).
    """
    import anndata as ad
    import scanpy as sc
    n_ref = counts.shape[0]
    scales, mean, var = normalized_gene_moments(counts)
    dn = seurat_dispersion_norm(mean, var)[0]
    hvg = np.sort(np.argsort(-np.where(np.isfinite(dn), dn, -np.inf), kind="stable")[:n_hvg])
    Hm = np.log1p((sp.diags(scales) @ counts[:, hvg].astype(np.float64)).toarray()).astype(np.float32)
    H = ad.AnnData(X=Hm)
    sc.pp.scale(H, max_value=10)
    sc.tl.pca(H, n_comps=n_pcs, svd_solver="arpack", random_state=seed)
    Pc = np.asarray(H.obsm["X_pca"], np.float64)
    Z = harmony(Pc, batch).astype(np.float64) if batch is not None and len(set(batch)) > 1 else Pc
    if labels is not None:
        codes, names = pool_labels(labels, min_type_cells)
        T = len(names)
        Dr = centroid_dist(Z, codes.astype(np.int64), T)
        return dict(codes=codes, names=names, n_t=np.bincount(codes, minlength=T), resolution=float("nan"), Z=Z,
                    D=(Dr / Dr.max()).astype(np.float32), D_raw=Dr.astype(np.float32))
    A = ad.AnnData(X=np.zeros((n_ref, 1), np.float32))
    A.obsm["X_h"] = Z
    sc.pp.neighbors(A, n_neighbors=n_neighbors, use_rep="X_h", random_state=seed)
    tried = {}

    def run(res):
        if res not in tried:
            sc.tl.leiden(A, resolution=res, key_added="lei", flavor="igraph", n_iterations=2, random_state=seed,
                         directed=False)
            tried[res] = A.obs["lei"].astype(int).values
            log(f"  leiden resolution {res:.4f} -> {int(tried[res].max()) + 1} clusters")
        return int(tried[res].max() + 1)

    t_lo, t_hi = t_range
    lo, hi = 0.05, 2.0
    for _ in range(14):
        mid = round((lo + hi) / 2, 4)
        T = run(mid)
        if T < t_lo:
            lo = mid
        elif T > t_hi:
            hi = mid
        else:
            break
    ok = {r: int(l.max() + 1) for r, l in tried.items() if t_lo <= l.max() + 1 <= t_hi}
    if ok:
        res = sorted(ok, key=lambda r: (abs(ok[r] - t_target), r))[0]
    else:   # the requested range was not reachable: largest number of clusters not above t_hi
        cand = {r: int(l.max() + 1) for r, l in tried.items() if l.max() + 1 <= t_hi}
        res = sorted(cand, key=lambda r: (-cand[r], r))[0]
        log(f"  warning: no resolution gave {t_lo}-{t_hi} clusters; using {cand[res]} at resolution {res}")
    lab = tried[res]
    n = np.bincount(lab)
    small = set(np.where(n < min_type_cells)[0].tolist())
    order = sorted([c for c in range(len(n)) if c not in small], key=lambda c: -n[c])
    code_of = {c: i for i, c in enumerate(order)}
    names = [f"L{i:02d}" for i in range(len(order))]
    if small:
        for c in small:
            code_of[int(c)] = len(order)
        names.append("small")
    codes = np.array([code_of[int(c)] for c in lab], np.int16)
    T = len(names)
    Dr = centroid_dist(Z, codes.astype(np.int64), T)
    D = Dr / Dr.max()
    return dict(codes=codes, names=names, n_t=np.bincount(codes, minlength=T), resolution=float(res), Z=Z,
                D=D.astype(np.float32), D_raw=Dr.astype(np.float32))
