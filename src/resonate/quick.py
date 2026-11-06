"""One-call helpers: ``impute`` (prepare + fit + predict) and ``score`` (per-gene accuracy of held-out genes)."""
from __future__ import annotations

import dataclasses

import numpy as np

from .config import Config


def impute(spatial, reference, gene_embeddings, genes=None, *, heldout=None, device=None, max_epochs=None,
           config: Config | None = None, return_model: bool = False, verbose: bool = True, **prepare_kwargs):
    """Predict genes in every spatial cell in one call: :func:`prepare`, :func:`fit`, then ``predict``.

    Parameters
    ----------
    spatial, reference, gene_embeddings
        As for :func:`prepare`: raw counts of the measured genes with ``obsm["spatial"]``; raw counts of a matched
        single-cell reference; one embedding per gene (path, table or :class:`GeneEmbeddings`).
    genes : list of str
        Genes the panel did not measure, to predict. Each must be in the reference and in the embedding table.
    heldout : list of str
        Measured panel genes to hide from the model and predict as well, to check accuracy with :func:`score`.
    device : str
        ``"cpu"`` or ``"cuda"`` (default: CUDA when available). ``"cpu"`` also switches off the GPU-only numerics.
    max_epochs : int
        Cap on the epochs of each training stage (default 500 with early stopping; 12 gives a run of minutes on a CPU
        for the demo data).
    config : Config
        Any other setting; ``device`` and ``max_epochs`` above override it.
    return_model : bool
        Also return the fitted :class:`Resonate` model (to save it or to predict more genes later).
    **prepare_kwargs
        Passed to :func:`prepare` (for example ``section_key``, ``batch_key``, ``spatial_layer``).

    Returns
    -------
    AnnData of the spatial cells x (``heldout`` + ``genes``), on the model's log scale (``log1p`` of normalised
    counts), with ``obs["split"]``, ``obsm["spatial"]`` and ``var["heldout"]``; with ``return_model=True``,
    ``(predictions, model)``.
    """
    from .api import fit
    from .prep.pipeline import prepare

    genes = [str(g) for g in (genes or [])]
    heldout = [str(g) for g in (heldout or [])]
    if not genes and not heldout:
        raise ValueError("give the genes to predict (genes=...) and/or measured genes to hold out (heldout=...)")
    measured = set(map(str, spatial.var_names))
    in_panel = [g for g in genes if g in measured]
    if in_panel:
        raise ValueError(f"{len(in_panel)} of genes= are measured in the panel (e.g. {in_panel[:5]}); they are model "
                         "inputs. Pass measured genes as heldout= to check how well they are predicted")
    missing = [g for g in heldout if g not in measured]
    if missing:
        raise ValueError(f"{len(missing)} of heldout= are not measured in the panel (e.g. {missing[:5]})")
    overrides = {}
    if device is not None:
        overrides["device"] = device
        if str(device) == "cpu":
            overrides.update(tf32=False, attn_bf16=False)
    if max_epochs is not None:
        overrides["max_epochs"] = int(max_epochs)
    cfg = dataclasses.replace(config or Config(), **overrides)
    prep = prepare(spatial, reference, gene_embeddings, target_genes=heldout + genes, config=cfg, verbose=verbose,
                   **prepare_kwargs)
    model = fit(prep, cfg, verbose=verbose)
    pred = model.predict()
    pred.var["heldout"] = [g in set(heldout) for g in pred.var_names]
    return (pred, model) if return_model else pred


def measured_log_expression(spatial, genes, layer=None):
    """Measured values of ``genes`` on the scale used for scoring: counts normalised by each cell's total count over
    the whole panel (scaled to the median total), then ``log1p``. Returns an array cells x genes."""
    import scipy.sparse as sp
    C = spatial.layers[layer] if layer else spatial.X
    C = sp.csr_matrix(C) if not sp.issparse(C) else C.tocsr()
    lib = np.asarray(C.sum(1)).ravel().astype(np.float64)
    target = float(np.median(lib[lib > 0]))
    scale = np.where(lib > 0, target / np.where(lib > 0, lib, 1.0), 0.0)
    cols = [spatial.var_names.get_loc(g) for g in genes]
    return np.log1p(C[:, cols].toarray().astype(np.float64) * scale[:, None])


def score(pred, spatial, genes=None, *, cells="test", top_n: int = 100, detection_floor: float = 0.10, layer=None):
    """Per-gene accuracy of predicted held-out genes against their measured values in ``spatial``.

    Parameters
    ----------
    pred : AnnData
        Predictions over the spatial cells (output of :func:`impute` or ``Resonate.predict``).
    spatial : AnnData
        The spatial data given to :func:`impute` (raw counts, including the held-out genes).
    genes : list of str
        Genes to score; default the held-out genes of ``pred`` (``var["heldout"]``), or every predicted gene that the
        panel measures.
    cells : str or array
        ``"test"`` (default: cells used neither for training nor for choosing the checkpoint), ``"val"``, ``"train"``,
        ``"all"``, or row positions.
    top_n, detection_floor
        Definition of the paper's headline gene set: genes detected in at least ``detection_floor`` of all cells,
        at most ``top_n`` of them (highest mean count first).

    Returns
    -------
    DataFrame indexed by gene and sorted by Pearson: ``pearson`` and ``spearman`` (across the scored cells, on the raw
    predictions), ``detected`` (fraction of all cells with a count) and ``top_n`` (in the headline set). The paper's
    headline number is ``table.loc[table.top_n, "pearson"].mean()``; it equals :func:`score_heldout`'s ``top_n``.
    """
    import pandas as pd

    from .metrics import score as _score

    if genes is None:
        if "heldout" in pred.var:
            genes = [g for g, h in zip(pred.var_names, pred.var["heldout"]) if h]
        else:
            genes = [g for g in pred.var_names if g in spatial.var_names]
    genes = [str(g) for g in genes]
    if not genes:
        raise ValueError("no held-out genes to score: pass genes= or predictions made with heldout=")
    sp_cells = spatial[pred.obs_names]
    truth = measured_log_expression(sp_cells, genes, layer)
    counts = sp_cells[:, genes].X
    counts = counts.toarray() if hasattr(counts, "toarray") else np.asarray(counts)
    detected = (counts > 0).mean(0)
    mean = counts.astype(np.float64).mean(0)
    order = np.argsort(-mean, kind="stable")
    headline = {genes[j] for j in [j for j in order if detected[j] >= detection_floor][:top_n]}
    if isinstance(cells, str):
        idx = np.arange(pred.n_obs) if cells == "all" else np.where(pred.obs["split"].astype(str).values == cells)[0]
    else:
        idx = np.asarray(cells)
    P = pred[:, genes].X
    P = P.toarray() if hasattr(P, "toarray") else np.asarray(P)
    s = _score(truth[idx], P[idx])
    table = pd.DataFrame({"pearson": s["per_gene"]["pearson"], "spearman": s["per_gene"]["spearman"],
                          "detected": detected, "top_n": [g in headline for g in genes]},
                         index=pd.Index(genes, name="gene"))
    return table.sort_values("pearson", ascending=False)
