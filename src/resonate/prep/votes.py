"""Reference lookups and cell-type mixtures, each computed under gene-fold exclusion (Methods, Training).

Measured genes are split at random into 20 folds. A target in fold f reads a reference lookup / type mixture computed
from the measured genes *outside* fold f (slot f); target genes and prediction time use slot 20, which uses all
measured genes. Searches are therefore repeated 21 times."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA

from ..model.model import DEPLOY_SLOT, K_NN, N_FOLDS
from .neighbors import (cosine_knn, harmony, joint_embed, knn_graph, leave_self_out_knn, maxfuse_smooth,
                        neighbor_mean, zscore_cols)


def gene_folds(n_measured: int, n_folds: int = N_FOLDS):
    """Random split of the measured genes (positions 0..n_measured-1) into folds; fixed seed 0."""
    rng = np.random.default_rng(0)
    return np.array_split(rng.permutation(n_measured), n_folds)


def gene_slots(G: int, vis_order: np.ndarray) -> np.ndarray:
    """(G,) int8 slot of each panel gene: its fold, or the deploy slot for genes that are not measured."""
    gs = np.full(G, DEPLOY_SLOT, np.int8)
    for fi, fl in enumerate(gene_folds(len(vis_order))):
        gs[vis_order[fl]] = fi
    return gs


def votes_from_idx(idx, codes, T):
    """(n, T) uint8 counts of the reference types among each row's neighbours."""
    out = np.zeros((idx.shape[0], T), np.uint8)
    for s in range(0, idx.shape[0], 50000):
        c = codes[idx[s:s + 50000]].astype(np.int64)
        b = c.shape[0]
        out[s:s + b] = np.bincount((np.arange(b)[:, None] * T + c).ravel(), minlength=b * T).reshape(b, T).astype(np.uint8)
    return out


def smoothing_and_lookup(ST, REF, vis_order, *, dev="cpu", wt=0.3, k_smooth=15, log=print):
    """Smoothed profiles, the spatial-to-reference lookups and the deploy neighbour sets.

    ST (N, G) spatial log-normalised values (target columns 0), REF (n_ref, G) reference panel values, vis_order the
    panel positions of the measured genes. Returns dict(nb_sp, nb_ref, STs (N, n_vis), REFs (n_ref, n_vis),
    nbr (N, 100) deploy reference neighbours, r_ref (N, G) lookup with all measured genes, r_ref_cv (N, G) gene-fold
    lookup (NaN in target columns))."""
    vis = np.asarray(vis_order, np.int64)
    N, G = ST.shape
    S_vis, R_vis = ST[:, vis], REF[:, vis]
    nb_sp = knn_graph(S_vis, k=k_smooth, dev=dev)
    nb_ref = knn_graph(R_vis, k=k_smooth, dev=dev)
    STs = maxfuse_smooth(S_vis, nb_sp, wt)
    REFs = maxfuse_smooth(R_vis, nb_ref, wt)
    del S_vis, R_vis
    st_e, rf_e = joint_embed(STs, REFs, seed=0)
    nbr = cosine_knn(st_e, rf_e, K_NN, dev)
    r_ref = neighbor_mean(nbr, REF)
    r_cv = np.full((N, G), np.nan, np.float32)
    for fi, fl in enumerate(gene_folds(len(vis))):
        keep = np.setdiff1d(np.arange(len(vis)), fl)
        st_e, rf_e = joint_embed(STs[:, keep], REFs[:, keep], seed=0)
        idx = cosine_knn(st_e, rf_e, K_NN, dev)
        r_cv[:, vis[fl]] = neighbor_mean(idx, REF[:, vis[fl]])
        log(f"  lookup fold {fi + 1}/{N_FOLDS}")
    return dict(nb_sp=nb_sp, nb_ref=nb_ref, STs=STs, REFs=REFs, nbr=nbr, r_ref=r_ref, r_ref_cv=r_cv)


def reference_votes(REFs, REF, vis_order, codes, T, batch=None, *, dev="cpu", log=print):
    """Leave-one-out type votes of every reference cell (Harmony-corrected by donor) and the reference-to-reference
    lookup used in pre-training, for all 21 gene-fold slots.

    Returns dict(counts (n_ref, 21, T) uint8, L_cv (n_ref, G) fold lookup, L_dep (n_ref, G) all-gene lookup)."""
    vis = np.asarray(vis_order, np.int64)
    n, G = REF.shape
    folds = gene_folds(len(vis))
    counts = np.zeros((n, N_FOLDS + 1, T), np.uint8)
    L_cv = np.full((n, G), np.nan, np.float32)
    L_dep = np.full((n, G), np.nan, np.float32)

    def nb_mean(idx, cols):
        rows = np.repeat(np.arange(n), idx.shape[1])
        S = sp.csr_matrix((np.full(idx.size, 1.0 / idx.shape[1], np.float32), (rows, idx.ravel())), shape=(n, n))
        return np.asarray(S @ REF[:, cols], np.float32)

    use_h = batch is not None and len(set(batch)) > 1
    for slot in list(range(N_FOLDS)) + [DEPLOY_SLOT]:
        keep = np.arange(len(vis)) if slot == DEPLOY_SLOT else np.setdiff1d(np.arange(len(vis)), folds[slot])
        E = PCA(n_components=min(50, len(keep)), random_state=0).fit_transform(zscore_cols(REFs[:, keep]))
        idx_u = leave_self_out_knn(E, K_NN, dev)            # the mean-term lookup is not batch corrected
        if slot == DEPLOY_SLOT:
            L_dep[:, :] = nb_mean(idx_u, np.arange(G))
        else:
            L_cv[:, vis[folds[slot]]] = nb_mean(idx_u, vis[folds[slot]])
        del idx_u
        idx = leave_self_out_knn(harmony(E, batch) if use_h else E, K_NN, dev)
        counts[:, slot] = votes_from_idx(idx, codes, T)
        log(f"  reference slot {slot + 1}/{N_FOLDS + 1}")
    fill = np.isnan(L_cv[0])
    L_cv[:, fill] = L_dep[:, fill]                          # target genes have no fold: deploy slot
    assert not np.isnan(L_cv).any()
    return dict(counts=counts, L_cv=L_cv, L_dep=L_dep)


def spatial_votes(STs, REFs, vis_order, codes, T, sections, batch=None, *, dev="cpu", log=print):
    """Type votes of every spatial cell: 100 nearest reference cells in the joint embedding after Harmony correction
    with spatial sections and reference donors as batches. (N, 21, T) uint8."""
    N = STs.shape[0]
    vis_n = STs.shape[1]
    folds = gene_folds(vis_n)
    ref_b = np.char.add("ref:", np.asarray(batch).astype(str)) if batch is not None else np.full(REFs.shape[0], "ref:all")
    b = np.concatenate([np.char.add("sp:", np.asarray(sections).astype(str)), ref_b])
    counts = np.zeros((N, N_FOLDS + 1, T), np.uint8)

    def run_slot(slot, st_in, rf_in):
        st_e, rf_e = joint_embed(st_in, rf_in, seed=0)
        Hz = harmony(np.vstack([st_e, rf_e]), b)
        idx = cosine_knn(Hz[:N], Hz[N:], K_NN, dev, chunk=1024)
        counts[:, slot] = votes_from_idx(idx, codes, T)
        log(f"  spatial slot {slot + 1}/{N_FOLDS + 1}")
    run_slot(DEPLOY_SLOT, STs, REFs)
    for fi, fl in enumerate(folds):
        keep = np.setdiff1d(np.arange(vis_n), fl)
        run_slot(fi, STs[:, keep], REFs[:, keep])
    return counts
