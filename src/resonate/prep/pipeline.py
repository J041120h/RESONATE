"""``prepare``: AnnData in, everything the model reads out (the "Model inputs" section of the Methods)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from ..config import Config
from ..embeddings import GeneEmbeddings, cosine_matrix, load_gene_embeddings
from ..model.model import DEPLOY_SLOT, K_NN
from .aux import build_aux
from .correlations import reference_split, type_correlations
from .neighbors import default_device, spatial_knn
from .reftypes import reference_types
from .votes import gene_slots, reference_votes, smoothing_and_lookup, spatial_votes

_ARRAYS = ["genes", "vis", "vis_order", "targets", "X", "X_input", "NB", "tr", "va", "te", "obs_names", "sections",
           "REF", "ref_obs_names", "ref_batch", "ref_tr", "ref_va", "ref_codes", "type_names", "pi", "type_dist",
           "A_t", "A_ref_tr", "votes_sp", "votes_ref", "gene_slot", "labels_sp", "L_cv", "r_ref", "r_ref_cv", "nbr",
           "V", "oov", "S_FM", "panel_library", "aux_counts_data", "aux_counts_indices", "aux_counts_indptr",
           "aux_counts_shape", "aux_symbols", "aux_V", "aux_A", "aux_panel_library", "truth_counts", "truth_genes",
           "lib_full", "ref_panel_cols", "xy"]


class Prepared:
    """Output of :func:`prepare`. Plain numpy arrays; ``save`` / ``load`` round-trip them.

    Gene axis G = measured genes (``vis``) followed by target genes (``targets``) unless a ``gene_order`` was given.
    Attributes are documented in ``docs/inputs-outputs.md``."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        arrays = {k: (v.astype(str) if getattr(v, 'dtype', None) == object else v)
                  for k, v in self.__dict__.items() if k in _ARRAYS and v is not None}
        np.savez(path / "arrays.npz", **arrays)
        json.dump(self.meta, open(path / "meta.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    @classmethod
    def load(cls, path):
        path = Path(path)
        z = np.load(path / "arrays.npz", allow_pickle=False)
        d = {k: z[k] for k in z.files}
        d["meta"] = json.load(open(path / "meta.json"))
        return cls(**d)

    @property
    def n_cells(self):
        return self.X.shape[0]

    @property
    def n_genes(self):
        return self.X.shape[1]

    def __repr__(self):
        return (f"Prepared({self.n_cells} spatial cells, {len(self.ref_codes)} reference cells, {len(self.vis)} measured "
                f"+ {len(self.targets)} target genes, {len(self.type_names)} reference types)")


def _get_matrix(adata, layer):
    X = adata.layers[layer] if layer else adata.X
    return sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()


def _check_counts(C, what):
    d = C.data
    if d.size and (d.min() < 0 or not np.allclose(d[:10000], np.round(d[:10000]), atol=1e-3)):
        raise ValueError(f"{what}: expected raw non-negative counts (use layer=... for the count layer)")


def prepare(spatial, reference, gene_embeddings, *, target_genes=None, heldout_fraction=None, heldout_seed=0,
            gene_order=None, spatial_key="spatial", section_key="section", batch_key="donor", split_key="split",
            label_key=None, type_key=None, spatial_layer=None, reference_layer=None, config: Config | None = None, verbose=True,
            out_dir=None) -> Prepared:
    """Build all model inputs from a spatial AnnData, a matched single-cell reference and gene embeddings.

    Parameters
    ----------
    spatial : AnnData
        Raw counts (``.X`` or ``layers[spatial_layer]``) of the measured genes; ``var_names`` = gene symbols;
        ``obsm[spatial_key]`` = coordinates; optional ``obs[section_key]`` (one value per tissue section; neighbours
        are searched within a section) and ``obs[split_key]`` in {train, val, test}.
    reference : AnnData
        Raw counts of a matched single-cell reference that measures the target genes; optional ``obs[batch_key]``
        (donor) used for batch correction.
    gene_embeddings : GeneEmbeddings, path, DataFrame, ...
        See :func:`resonate.embeddings.load_gene_embeddings`.
    target_genes : list of str
        Genes to predict. Targets that are also in ``spatial.var_names`` are *held out* (removed from the model input
        before normalisation; their measured values are kept only for evaluation).
    heldout_fraction : float
        Alternative to ``target_genes`` for benchmarking: hold out this fraction of the measured genes, drawn with
        ``numpy.random.default_rng(heldout_seed)``.
    type_key : str, optional
        Column of ``reference.obs`` with cell types to use instead of Leiden clusters of the reference (types with fewer than
        ``min_type_cells`` cells are pooled). Default (``None``): Leiden clusters, as in the paper.
    gene_order : list of str, optional
        Explicit order of the model's gene axis (measured and target genes); default measured then targets.
    """
    t0 = time.time()
    cfg = config or Config()
    dev = cfg.device or default_device()
    log = (lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)) if verbose else (lambda *a: None)
    emb = load_gene_embeddings(gene_embeddings)

    # ---- genes ------------------------------------------------------------------------------------------------------
    measured = [str(g) for g in spatial.var_names]
    if len(set(measured)) != len(measured):
        raise ValueError("spatial.var_names must be unique")
    if len(set(reference.var_names)) != reference.n_vars:
        raise ValueError("reference.var_names must be unique (make them unique or aggregate duplicate symbols)")
    if (target_genes is None) == (heldout_fraction is None):
        raise ValueError("give exactly one of target_genes or heldout_fraction")
    if heldout_fraction is not None:
        k = int(np.ceil(heldout_fraction * len(measured)))
        pick = np.random.default_rng(heldout_seed).choice(len(measured), size=k, replace=False)
        target_genes = [measured[i] for i in sorted(pick)]
    targets = [str(g) for g in target_genes]
    if len(set(targets)) != len(targets):
        raise ValueError("target_genes contains duplicates")
    held = [g for g in targets if g in set(measured)]
    vis_names = [g for g in measured if g not in set(targets)]
    if len(vis_names) < 20:
        raise ValueError(f"only {len(vis_names)} measured genes remain after holding out the targets")
    ref_names = [str(g) for g in reference.var_names]
    missing = [g for g in vis_names + targets if g not in set(ref_names)]
    if missing:
        raise ValueError(f"{len(missing)} panel / target genes are absent from the reference, e.g. {missing[:5]}")
    genes = vis_names + [g for g in targets]
    if gene_order is not None:
        if sorted(gene_order) != sorted(genes):
            raise ValueError("gene_order must contain exactly the measured (non-target) and target genes")
        genes = [str(g) for g in gene_order]
    G = len(genes)
    pos = {g: i for i, g in enumerate(genes)}
    vis_order = np.array([pos[g] for g in vis_names], np.int64)       # panel position of each measured gene
    tgt_idx = np.array([pos[g] for g in targets], np.int64)
    log(f"panel {G} genes = {len(vis_names)} measured + {len(targets)} targets ({len(held)} held out from the input)")

    # ---- spatial input: library size over measured genes only ------------------------------------------------------
    Cs = _get_matrix(spatial, spatial_layer)
    _check_counts(Cs, "spatial")
    scol = {g: i for i, g in enumerate(measured)}
    C_vis = Cs[:, [scol[g] for g in vis_names]].astype(np.float64)
    lib = np.asarray(C_vis.sum(1)).ravel()
    target_sum = float(np.median(lib[lib > 0]))
    Y = (sp.diags(np.where(lib > 0, target_sum / np.where(lib > 0, lib, 1.0), 0.0)) @ C_vis).tocsr()
    Y.data = np.log1p(Y.data)
    N = spatial.n_obs
    X = np.zeros((N, G), np.float32)
    X[:, vis_order] = Y.toarray().astype(np.float32)
    del Y, C_vis
    # ---- evaluation truth (never read by training) ----------------------------------------------------------------
    truth_counts = truth_genes = lib_full = None
    if held:
        truth_genes = np.array(held)
        truth_counts = Cs[:, [scol[g] for g in held]].toarray().astype(np.float32)
        lib_full = np.asarray(Cs.sum(1)).ravel().astype(np.float64)
    del Cs

    # ---- splits, sections, coordinates ------------------------------------------------------------------------------
    if split_key in spatial.obs:
        split = spatial.obs[split_key].astype(str).values
        tr, va, te = (np.where(split == p)[0].astype(np.int64) for p in ("train", "val", "test"))
        if len(tr) == 0 or len(va) == 0:
            raise ValueError(f"obs[{split_key!r}] must contain 'train' and 'val' (and 'test') cells")
    else:
        rng = np.random.default_rng(cfg.seed)
        strat = spatial.obs[label_key].astype(str).values if label_key else np.zeros(N, str)
        tr_l, va_l, te_l = [], [], []
        for u in np.unique(strat):
            rows = rng.permutation(np.where(strat == u)[0])
            n_te = int(round(cfg.split_fractions[2] * len(rows)))
            n_va = int(round(cfg.split_fractions[1] * len(rows)))
            te_l.append(rows[:n_te]); va_l.append(rows[n_te:n_te + n_va]); tr_l.append(rows[n_te + n_va:])
        tr, va, te = (np.sort(np.concatenate(a)).astype(np.int64) for a in (tr_l, va_l, te_l))
    sections = spatial.obs[section_key].astype(str).values if section_key in spatial.obs else np.full(N, "all")
    if spatial_key not in spatial.obsm:
        raise ValueError(f"spatial.obsm[{spatial_key!r}] (cell coordinates) is required")
    xy = np.asarray(spatial.obsm[spatial_key], np.float64)[:, :2]
    NB = spatial_knn(xy, sections, cfg.n_spatial_nb)

    # ---- reference ------------------------------------------------------------------------------------------------
    Cr = _get_matrix(reference, reference_layer)
    _check_counts(Cr, "reference")
    rpos = {g: i for i, g in enumerate(ref_names)}
    rcol = np.array([rpos[g] for g in genes])
    Cp = Cr[:, rcol].astype(np.float64)
    plib = np.asarray(Cp.sum(1)).ravel()
    Yr = (sp.diags(np.where(plib > 0, 1e4 / np.where(plib > 0, plib, 1.0), 0.0)) @ Cp).tocsr()
    Yr.data = np.log1p(Yr.data)
    REF = Yr.toarray().astype(np.float32)
    del Yr, Cp
    batch = reference.obs[batch_key].astype(str).values if batch_key in reference.obs else None
    log(f"{N} spatial cells ({len(tr)}/{len(va)}/{len(te)} train/val/test), {REF.shape[0]} reference cells, "
        f"{len(set(batch)) if batch is not None else 1} reference batch(es), device {dev}")

    # ---- smoothing, lookups -----------------------------------------------------------------------------------------
    sm = smoothing_and_lookup(X, REF, vis_order, dev=dev, wt=cfg.smooth_weight, k_smooth=cfg.k_smooth, log=log)
    X_input = X.copy()
    X_input[:, vis_order] = sm["STs"]

    # ---- reference cell types ---------------------------------------------------------------------------------------
    log("reference cell types (" + (f"given: obs[{type_key!r}]" if type_key else "Leiden on Harmony-corrected reference") + ")")
    rt = reference_types(Cr, batch, labels=reference.obs[type_key].astype(str).values if type_key else None, t_range=cfg.n_types_range, t_target=cfg.n_types_target,
                         min_type_cells=cfg.min_type_cells, log=log)
    codes, names = rt["codes"], rt["names"]
    T = len(names)
    pi = rt["n_t"] / REF.shape[0]
    ref_tr, ref_va = reference_split(REF.shape[0])
    log("type-specific correlations")
    cc = type_correlations(REF, codes, names, ref_tr, min_type_cells=cfg.min_type_cells, shrink_n0=cfg.shrink_n0)

    # ---- votes ------------------------------------------------------------------------------------------------------
    log("reference votes")
    rv = reference_votes(sm["REFs"], REF, vis_order, codes, T, batch, dev=dev, log=log)
    log("spatial votes")
    votes_sp = spatial_votes(sm["STs"], sm["REFs"], vis_order, codes, T, sections, batch, dev=dev, log=log)
    labels_sp = np.argmax(votes_sp[:, DEPLOY_SLOT].astype(np.float64) + pi, 1).astype(np.int64)
    slots = gene_slots(G, vis_order)
    assert (slots[tgt_idx] == DEPLOY_SLOT).all() and (votes_sp.sum(2) == K_NN).all()

    # ---- embeddings and auxiliary targets ---------------------------------------------------------------------------
    V, oov = emb.panel(genes)
    S_FM = cosine_matrix(V, oov)
    log(f"embeddings: {int(oov.sum())} of {G} panel genes have none (they share one learnable vector)")
    log("auxiliary targets")
    ax = build_aux(Cr, ref_names, genes, rcol, REF, emb, n_pool=cfg.n_aux_genes)
    P = dict(genes=np.array(genes), vis=np.sort(vis_order), vis_order=vis_order, targets=tgt_idx, X=X, X_input=X_input,
             NB=NB, tr=tr, va=va, te=te, obs_names=np.array(spatial.obs_names.astype(str)), sections=sections,
             REF=REF, ref_obs_names=np.array(reference.obs_names.astype(str)),
             ref_batch=batch if batch is not None else np.array([], str), ref_tr=ref_tr, ref_va=ref_va, ref_codes=codes,
             type_names=np.array(names), pi=pi, type_dist=rt["D"], A_t=cc["A_t"], A_ref_tr=cc["A_ref_tr"],
             votes_sp=votes_sp, votes_ref=rv["counts"], gene_slot=slots, labels_sp=labels_sp, L_cv=rv["L_cv"],
             r_ref=sm["r_ref"], r_ref_cv=sm["r_ref_cv"], nbr=sm["nbr"], V=V, oov=oov, S_FM=S_FM, panel_library=plib,
             aux_counts_data=ax["counts"].data, aux_counts_indices=ax["counts"].indices.astype(np.int32),
             aux_counts_indptr=ax["counts"].indptr.astype(np.int64), aux_counts_shape=np.array(ax["counts"].shape),
             aux_symbols=ax["pool_symbols"], aux_V=ax["V_aux"], aux_A=ax["A_aux"], aux_panel_library=ax["panel_library"],
             truth_counts=truth_counts, truth_genes=truth_genes, lib_full=lib_full, ref_panel_cols=rcol,
             xy=xy.astype(np.float32))
    meta = dict(n_cells=int(N), n_ref=int(REF.shape[0]), G=int(G), n_measured=int(len(vis_names)), n_targets=int(len(targets)),
                n_heldout=int(len(held)), T=int(T), spatial_target_sum=target_sum, ref_resolution=rt["resolution"], type_source=(type_key or "leiden"),
                n_t=rt["n_t"].tolist(), config=cfg.to_dict(), minutes=(time.time() - t0) / 60,
                aux_target_sum=ax["target_sum"], nbr_note="nbr = 100 deploy-slot reference neighbours of each spatial cell")
    log(f"prepared in {meta['minutes']:.1f} min")
    prep = Prepared(meta=meta, **P)
    if out_dir:
        prep.save(out_dir)
    return prep
