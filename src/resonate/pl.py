"""Plots of inputs and predictions (matplotlib; ``pip install matplotlib`` or the ``[notebook]`` extra).

Every function shows its figure (``show=True``, as in a notebook) or, with ``show=False``, returns the matplotlib
``Figure`` to save or change it. Maps draw the cells at their ``obsm["spatial"]`` coordinates with a
point size chosen from the number of cells and the area of the section (``size=`` overrides it).
"""
from __future__ import annotations

import numpy as np

TEAL = "#0b6f5d"
GREY = "#d5dadf"
SLATE = "#33485d"
_RAMP = ["#eef2f3", "#9fdcc9", "#10b396", "#0b6f5d", "#0b3b35"]


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as err:          # pragma: no cover
        raise ImportError("resonate.pl needs matplotlib: pip install matplotlib") from err
    return plt


def _finish(fig, show):
    plt = _plt()
    if show:
        plt.show()
        return None
    plt.close(fig)
    return fig


def _cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("resonate", _RAMP)


def _categories(column):
    """Category order of a pandas column (sorted unique values if it is not categorical)."""
    if hasattr(column, "cat"):
        return [str(c) for c in column.cat.categories]
    return sorted(set(column.astype(str)))


def _point_size(xy, panel_inches):
    """Marker area (points^2) so that the cells roughly tile the occupied part of the panel."""
    span = max(np.ptp(xy[:, 0]), np.ptp(xy[:, 1]), 1e-9)
    gx, gy = np.floor((xy - xy.min(0)) / (span / 50)).T                 # fraction of a 50 x 50 grid with cells
    occupied = len(set(zip(gx.tolist(), gy.tolist()))) / 2500
    diameter = np.sqrt(occupied * span ** 2 / len(xy)) / span * panel_inches * 72
    return float(np.clip(0.9 * diameter ** 2, 0.2, 30))


def _map_axes(n, ncols, panel, nrows=None):
    plt = _plt()
    nrows = nrows or int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(panel * ncols, panel * nrows * 1.0), squeeze=False)
    for ax in axes.ravel():
        ax.set_aspect("equal")
        ax.axis("off")
    return fig, axes


def _colorbar(fig, mappable, ax):
    bar = fig.colorbar(mappable, cax=ax.inset_axes([0.25, -0.05, 0.5, 0.03]), orientation="horizontal")
    bar.ax.tick_params(labelsize=7)


def cell_types(spatial, key="cell_group", groups=None, *, ncols=4, panel=3.0, size=None, show=True):
    """One map per cell type of ``spatial.obs[key]``: the cells of that type in colour, the others in grey.

    ``groups`` selects and orders the types (default: all categories except ``"Other"``)."""
    labels = spatial.obs[key].astype(str).to_numpy()
    if groups is None:
        groups = [g for g in _categories(spatial.obs[key]) if g != "Other"]
    xy = np.asarray(spatial.obsm["spatial"])[:, :2]
    s = size or _point_size(xy, panel)
    fig, axes = _map_axes(len(groups), ncols, panel)
    for ax, g in zip(axes.ravel(), groups):
        m = labels == g
        ax.scatter(xy[~m, 0], xy[~m, 1], s=s * 0.6, c=GREY, linewidths=0, rasterized=True)
        ax.scatter(xy[m, 0], xy[m, 1], s=s, c=TEAL, linewidths=0, rasterized=True)
        ax.set_title(f"{g} ({m.sum():,})", fontsize=10)
    for ax in axes.ravel()[len(groups):]:
        ax.set_visible(False)
    fig.tight_layout()
    return _finish(fig, show)


def gene_scores(table, *, figsize=(11, 3.0), show=True):
    """Bar plot of the per-gene Pearson correlation from :func:`resonate.score`; genes outside the headline set
    (``top_n``) in grey, the dashed line is the mean over the headline set."""
    plt = _plt()
    t = table.sort_values("pearson", ascending=False)
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(t)), t["pearson"], width=0.75, color=[TEAL if h else "#9aa5b1" for h in t["top_n"]])
    mean = t.loc[t["top_n"], "pearson"].mean()
    ax.axhline(mean, color=SLATE, lw=1, ls="--")
    ax.text(len(t) - 0.5, mean + 0.02, f"mean {mean:.3f}", ha="right", va="bottom", fontsize=8, color=SLATE)
    ax.set_xticks(range(len(t)), t.index, rotation=90, fontsize=7)
    ax.set_xlim(-0.6, len(t) - 0.4)
    ax.set_ylabel("Pearson r (test cells)")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _finish(fig, show)


def spatial_compare(pred, spatial, genes, *, table=None, panel=3.0, size=None, show=True):
    """Measured (top row) and predicted (bottom row) maps of held-out genes.

    Predictions are shown rank-matched to the measured values of the same gene: the cell with the k-th highest
    prediction gets the k-th highest measured value. Measured and predicted maps then share one colour scale, clipped to
    the 1st-99th percentile of the measured values. Titles give the Pearson r of the raw predictions in the test cells
    (from ``table``, the output of :func:`resonate.score`, computed here if not given)."""
    from .quick import measured_log_expression, score
    genes = [str(g) for g in genes]
    table = score(pred, spatial, genes) if table is None else table
    sp_cells = spatial[pred.obs_names]
    truth = measured_log_expression(sp_cells, genes)
    P = pred[:, genes].X
    P = P.toarray() if hasattr(P, "toarray") else np.asarray(P)
    xy = np.asarray(pred.obsm["spatial"])[:, :2]
    s = size or _point_size(xy, panel)
    cmap = _cmap()
    fig, axes = _map_axes(2 * len(genes), len(genes), panel, nrows=2)
    for j, g in enumerate(genes):
        meas = truth[:, j]
        matched = np.empty_like(meas)
        matched[np.argsort(P[:, j], kind="stable")] = np.sort(meas)
        lo, hi = np.percentile(meas, [1, 99])
        for i, (vals, label) in enumerate([(meas, "measured"),
                                           (matched, f"predicted, r = {table.loc[g, 'pearson']:.2f}")]):
            order = np.argsort(vals, kind="stable")
            dots = axes[i, j].scatter(xy[order, 0], xy[order, 1], c=vals[order], s=s, cmap=cmap, vmin=lo, vmax=hi,
                                      linewidths=0, rasterized=True)
            axes[i, j].set_title(f"{g}, {label}", fontsize=10)
        _colorbar(fig, dots, axes[1, j])
    fig.tight_layout(h_pad=1.5)
    return _finish(fig, show)


def spatial_genes(pred, genes, *, ncols=4, panel=3.0, size=None, show=True):
    """Maps of predicted genes (for example genes outside the panel), each coloured from its 1st to its 99th
    percentile."""
    genes = [str(g) for g in genes]
    P = pred[:, genes].X
    P = P.toarray() if hasattr(P, "toarray") else np.asarray(P)
    xy = np.asarray(pred.obsm["spatial"])[:, :2]
    s = size or _point_size(xy, panel)
    cmap = _cmap()
    fig, axes = _map_axes(len(genes), ncols, panel)
    for ax, (j, g) in zip(axes.ravel(), enumerate(genes)):
        v = P[:, j]
        lo, hi = np.percentile(v, [1, 99])
        order = np.argsort(v, kind="stable")
        dots = ax.scatter(xy[order, 0], xy[order, 1], c=v[order], s=s, cmap=cmap, vmin=lo, vmax=hi, linewidths=0,
                          rasterized=True)
        ax.set_title(f"{g}, predicted", fontsize=10)
        _colorbar(fig, dots, ax)
    for ax in axes.ravel()[len(genes):]:
        ax.set_visible(False)
    fig.tight_layout()
    return _finish(fig, show)


def _scaled_means(X, labels, groups):
    import pandas as pd
    df = pd.DataFrame(X).groupby(np.asarray(labels).astype(str)).mean().loc[groups].T
    lo, hi = df.min(axis=1), df.max(axis=1)
    return df.sub(lo, axis=0).div((hi - lo).replace(0, 1), axis=0)


def celltype_heatmap(pred, spatial, groupby, genes, *, groups=None, reference=None, reference_groupby=None,
                     reference_groups=None, figsize=(10, 4.6), show=True):
    """Mean predicted expression of ``genes`` in each group of spatial cells (``spatial.obs[groupby]``); with a
    ``reference``, next to the mean measured expression in each of its cell types (``log1p`` of counts per 10,000).
    Each row is scaled from its lowest (light) to its highest (dark) group, because the two sides use different scales."""
    plt = _plt()
    genes = [str(g) for g in genes]
    labels = spatial[pred.obs_names].obs[groupby].astype(str).to_numpy()
    groups = groups or [g for g in _categories(spatial.obs[groupby]) if g != "Other" and g in set(labels)]
    P = pred[:, genes].X
    P = P.toarray() if hasattr(P, "toarray") else np.asarray(P)
    panels = [("Predicted, spatial cells", _scaled_means(P, labels, groups))]
    if reference is not None:
        import scipy.sparse as sp
        C = sp.csr_matrix(reference.X) if not sp.issparse(reference.X) else reference.X.tocsr()
        lib = np.asarray(C.sum(1)).ravel()
        R = np.log1p(C[:, [reference.var_names.get_loc(g) for g in genes]].toarray()
                     * np.where(lib > 0, 1e4 / np.where(lib > 0, lib, 1), 0)[:, None])
        rlab = reference.obs[reference_groupby].astype(str).to_numpy()
        rgroups = reference_groups or list(np.unique(rlab))
        panels.append(("Measured, reference cells", _scaled_means(R, rlab, rgroups)))
    fig, axes = plt.subplots(1, len(panels), figsize=figsize, squeeze=False,
                             gridspec_kw={"width_ratios": [p[1].shape[1] for p in panels]})
    for ax, (title, df) in zip(axes[0], panels):
        ax.imshow(df.to_numpy(), cmap=_cmap(), vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(df.shape[1]), df.columns, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(genes)), genes if ax is axes[0, 0] else [""] * len(genes), fontsize=8)
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return _finish(fig, show)
