"""Scoring of held-out genes (Methods, "Benchmark design and scoring")."""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

EPS_STD = 1e-8


def per_gene_pearson(true, pred, min_std=1e-6):
    """Pearson correlation per column; NaN where either column is (near) constant."""
    t = np.asarray(true, np.float64)
    p = np.asarray(pred, np.float64)
    tc, pc = t - t.mean(0), p - p.mean(0)
    ts, ps = t.std(0), p.std(0)
    den = np.sqrt((tc ** 2).sum(0) * (pc ** 2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = (tc * pc).sum(0) / den
    return np.where((ts > min_std) & (ps > min_std) & (den > 0), r, np.nan)


def truth_log_expression(prepared):
    """Held-out truth on the log scale: counts normalised by the full-panel library (median target), log1p."""
    if prepared.truth_counts is None:
        raise ValueError("no held-out genes with measured values in this prepared object")
    lib = prepared.lib_full
    target = float(np.median(lib[lib > 0]))
    scale = np.where(lib > 0, target / np.where(lib > 0, lib, 1.0), 0.0)
    return np.log1p(prepared.truth_counts.astype(np.float64) * scale[:, None])


def top_n_genes(prepared, top_n=100, detection_floor=0.10):
    """Positions (in ``prepared.truth_genes``) of the scored genes: held-out genes detected in >= 10 % of all spatial
    cells, limited to the ``top_n`` with the highest mean count."""
    C = prepared.truth_counts
    det = (C > 0).mean(0)
    mean = C.astype(np.float64).mean(0)
    order = np.argsort(-mean, kind="stable")
    keep = [j for j in order if det[j] >= detection_floor][:top_n]
    return np.array(sorted(keep), dtype=int)


def score(truth, pred):
    """Pearson, Spearman and aligned MSE averaged over genes. Degenerate genes (constant truth or prediction) score
    0 and stay in the mean. truth, pred: (cells, genes)."""
    truth = np.asarray(truth, np.float64)
    pred = np.asarray(pred, np.float64)
    t_std = truth.std(0)
    p_std = pred.std(0)
    degen = (t_std <= EPS_STD) | (p_std <= EPS_STD) | ~np.isfinite(pred).all(0)
    r = np.nan_to_num(per_gene_pearson(truth, pred, min_std=EPS_STD), nan=0.0)
    r[degen] = 0.0
    rt = np.apply_along_axis(rankdata, 0, truth)
    rp = np.apply_along_axis(rankdata, 0, pred)
    rho = np.nan_to_num(per_gene_pearson(rt, rp, min_std=EPS_STD), nan=0.0)
    rho[degen] = 0.0
    t_mu, p_mu = truth.mean(0), pred.mean(0)
    vp = pred.var(0)
    a = np.where(vp > 0, np.maximum(0.0, ((pred - p_mu) * (truth - t_mu)).mean(0) / np.where(vp > 0, vp, 1.0)), 0.0)
    a[degen] = 0.0
    aligned = pred * a + (t_mu - a * p_mu)
    mse = ((truth - aligned) ** 2).mean(0)
    return dict(pearson=float(r.mean()), spearman=float(rho.mean()), mse=float(mse.mean()), n_genes=int(truth.shape[1]),
                n_cells=int(truth.shape[0]), n_degenerate=int(degen.sum()),
                per_gene=dict(pearson=r.tolist(), spearman=rho.tolist(), mse=mse.tolist()))


def score_heldout(prepared, pred, genes=None, *, cells="test", top_n=100, detection_floor=0.10):
    """Score predictions of held-out genes against their measured values.

    ``pred`` is (cells in ``cells``, len(genes)) or an AnnData over all spatial cells; ``genes`` names its columns.
    Returns the scores on the top-N genes (the headline number) and on all held-out genes with a measured value."""
    cell_idx = {"test": prepared.te, "val": prepared.va, "train": prepared.tr, "all": np.arange(prepared.n_cells)}[cells] \
        if isinstance(cells, str) else np.asarray(cells)
    if hasattr(pred, "X"):
        genes = list(pred.var_names)
        pred = pred.X.toarray() if hasattr(pred.X, "toarray") else np.asarray(pred.X)
    pred = np.asarray(pred)
    if pred.shape[0] == prepared.n_cells:
        pred = pred[cell_idx]
    T = truth_log_expression(prepared)[cell_idx]
    col = {g: j for j, g in enumerate(prepared.truth_genes)}
    use = [i for i, g in enumerate(genes) if g in col]
    if not use:
        raise ValueError("none of the predicted genes has a measured value in the spatial data")
    tcols = np.array([col[genes[i]] for i in use])
    top = top_n_genes(prepared, top_n, detection_floor)
    sel = np.array([k for k, c in enumerate(tcols) if c in set(top.tolist())])
    out = dict(all_heldout=score(T[:, tcols], pred[:, use]))
    if len(sel):
        out["top_n"] = score(T[:, tcols[sel]], pred[:, np.array(use)[sel]])
        out["top_n_genes"] = [genes[use[k]] for k in sel]
    return out
