"""Nearest-neighbour searches, smoothing and the joint embeddings used by the preparation step."""
from __future__ import annotations

import numpy as np
import torch


def default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def zscore_cols(M: np.ndarray) -> np.ndarray:
    """Column z-score with the +1e-8 stabiliser used throughout."""
    mu = M.mean(0, keepdims=True)
    sd = M.std(0, keepdims=True) + 1e-8
    return ((M - mu) / sd).astype(np.float32)


def knn_graph(M: np.ndarray, k: int = 15, n_pca: int = 30, dev: str = "cpu", chunk: int = 4096) -> np.ndarray:
    """(n, k) cosine nearest neighbours (self excluded) in the top ``n_pca`` PCs of the z-scored columns of ``M``."""
    from sklearn.decomposition import PCA
    E = PCA(n_components=min(n_pca, M.shape[1]), random_state=0).fit_transform(zscore_cols(M))
    e = torch.nn.functional.normalize(torch.tensor(E, dtype=torch.float32), dim=1).to(dev)
    out = []
    for i in range(0, e.shape[0], chunk):
        s = e[i:i + chunk] @ e.T
        s[torch.arange(s.shape[0]), torch.arange(i, i + s.shape[0])] = -2
        out.append(torch.topk(s, k, dim=1).indices.cpu().numpy())
    return np.concatenate(out, 0).astype(np.int32)


def maxfuse_smooth(M: np.ndarray, nb: np.ndarray, wt: float = 0.3) -> np.ndarray:
    """x~ = wt * x + (1 - wt) * mean of the graph neighbours (MaxFuse graph smoothing; Methods equation 2)."""
    out = np.empty_like(M, dtype=np.float32)
    for i in range(0, M.shape[0], 20000):
        out[i:i + 20000] = wt * M[i:i + 20000] + (1 - wt) * M[nb[i:i + 20000]].mean(1)
    return out


def joint_embed(ST_vis: np.ndarray, REF_vis: np.ndarray, n_pca: int = 50, seed: int = 0):
    """Per-dataset z-scored measured genes, joint PCA fitted on both datasets; returns both projections."""
    from sklearn.decomposition import PCA
    Z = np.vstack([zscore_cols(ST_vis), zscore_cols(REF_vis)])
    p = PCA(n_components=min(n_pca, Z.shape[1]), random_state=seed).fit(Z)
    return p.transform(zscore_cols(ST_vis)), p.transform(zscore_cols(REF_vis))


def cosine_knn(query: np.ndarray, ref: np.ndarray, K: int, dev: str = "cpu", chunk: int = 4096) -> np.ndarray:
    """Top-K cosine neighbours of each query row among the ref rows. (n_query, K) int32."""
    st = torch.nn.functional.normalize(torch.tensor(query, dtype=torch.float32), dim=1)
    rf = torch.nn.functional.normalize(torch.tensor(ref, dtype=torch.float32), dim=1).to(dev)
    out = []
    for i in range(0, st.shape[0], chunk):
        _, ix = torch.topk(st[i:i + chunk].to(dev) @ rf.t(), K, dim=1, largest=True)
        out.append(ix.cpu().numpy().astype(np.int32))
    return np.concatenate(out, 0)


def neighbor_mean(idx: np.ndarray, REF_cols: np.ndarray) -> np.ndarray:
    """Mean over the K neighbours of reference values (n_ref, m); chunked so the (n, K, m) gather never exists."""
    N, K = idx.shape
    m = REF_cols.shape[1]
    out = np.empty((N, m), np.float32)
    step = max(256, int(2.5e8 // max(1, K * m)))
    for s in range(0, N, step):
        e = min(s + step, N)
        out[s:e] = REF_cols[idx[s:e]].mean(1)
    return out


def leave_self_out_knn(E: np.ndarray, K: int = 100, dev: str = "cpu") -> np.ndarray:
    """K cosine neighbours of every row among the other rows (the row itself is dropped)."""
    e = torch.nn.functional.normalize(torch.tensor(E, dtype=torch.float32), dim=1)
    et = e.to(dev)
    n = e.shape[0]
    idx = np.empty((n, K), np.int32)
    for i in range(0, n, 1024):
        top = torch.topk(et[i:i + 1024] @ et.T, K + 1, dim=1).indices.cpu().numpy()
        selfcol = top == np.arange(i, i + top.shape[0])[:, None]
        keep = ~selfcol
        keep[~selfcol.any(1), K] = False            # self absent from the top K + 1: drop the last candidate
        idx[i:i + top.shape[0]] = top[keep].reshape(top.shape[0], K)
    return idx


def spatial_knn(xy: np.ndarray, sections: np.ndarray, k: int = 15) -> np.ndarray:
    """k nearest spatial neighbours of every cell within its own section, ordered by distance, self excluded."""
    from sklearn.neighbors import NearestNeighbors
    out = np.zeros((xy.shape[0], k), np.int64)
    for s in np.unique(sections):
        rows = np.where(sections == s)[0]
        if len(rows) <= k:
            raise ValueError(f"section {s!r} has only {len(rows)} cells; need more than k={k}")
        ind = NearestNeighbors(n_neighbors=k + 1).fit(xy[rows]).kneighbors(xy[rows], return_distance=False)[:, 1:]
        out[rows] = rows[ind]
    return out


def harmony(E: np.ndarray, batch) -> np.ndarray:
    """Harmony batch correction (harmonypy, random_state 0)."""
    import logging
    import harmonypy
    import pandas as pd
    logging.getLogger("harmonypy").setLevel(logging.WARNING)
    Z = np.asarray(harmonypy.run_harmony(np.asarray(E, np.float64), pd.DataFrame({"b": np.asarray(batch).astype(str)}),
                                         ["b"], random_state=0).Z_corr, np.float64)
    return (Z.T if Z.shape[0] != E.shape[0] else Z).astype(np.float32)
