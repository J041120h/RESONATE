"""Public workflow: ``fit`` a model on prepared inputs, ``predict`` unmeasured genes, save / load."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

from .config import Config
from .embeddings import load_gene_embeddings
from .model.model import ResonateNet, TargetBlock, negative_sets, positive_sets
from .prep.correlations import cross_corr
from .prep.neighbors import default_device, neighbor_mean
from .prep.pipeline import Prepared
from .train import build_net, partner_sets, predict_cells, set_value_stats, stage1, stage2, to_dev


def _setup_torch(cfg, device):
    if str(device).startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.tf32)


class Resonate:
    """A fitted model bound to the :class:`~resonate.prep.pipeline.Prepared` inputs it was trained on."""

    def __init__(self, net: ResonateNet, prepared: Prepared, cfg: Config, summary: dict, history: dict, device: str):
        self.net, self.prepared, self.cfg, self.summary, self.history, self.device = net, prepared, cfg, summary, history, device

    # ------------------------------------------------------------------------------------------------------------------
    @property
    def target_genes(self):
        return [str(self.prepared.genes[i]) for i in self.prepared.targets]

    def predict(self, genes=None, *, cells=None, reference=None, gene_embeddings=None, as_anndata=True, batch_size=512,
                gene_chunk=128):
        """Predict log-normalised expression of ``genes`` in the spatial cells.

        Genes that were targets when the model was fitted (``self.target_genes``) need nothing else. Any other gene
        measured in the reference (and present in the embedding table) is predicted without refitting: pass the
        ``reference`` AnnData used in :func:`prepare` and the ``gene_embeddings``.

        Returns an AnnData (cells x genes; ``X`` on the model's log scale, i.e. log1p of counts normalised to the median
        measured-gene library) or, with ``as_anndata=False``, ``(array, genes)``. ``cells`` = row indices (default: all).
        """
        prep, net, dev = self.prepared, self.net, self.device
        genes = self.target_genes if genes is None else [str(g) for g in genes]
        cells = np.arange(prep.n_cells) if cells is None else np.asarray(cells)
        tg = {g: j for j, g in enumerate(self.target_genes)}
        measured = {str(prep.genes[i]) for i in prep.vis}
        used = sorted(g for g in genes if g in measured)
        if used:
            raise ValueError(f"{len(used)} requested genes are measured in the spatial panel and are model inputs (e.g. "
                             f"{used[:5]}); predict unmeasured genes, or hold the gene out with prepare(target_genes=...)")
        panel_pos = {str(g): i for i, g in enumerate(prep.genes)}
        out = np.zeros((len(cells), len(genes)), np.float32)
        in_panel = [k for k, g in enumerate(genes) if g in tg]
        extra = [k for k, g in enumerate(genes) if g not in tg]
        if in_panel:
            G = net.G
            full = np.zeros(G, bool)
            full[prep.vis] = True
            P = predict_cells(net, cells, lambda i, n: np.broadcast_to(full, (n, G)).copy(), dev, "test", bs=batch_size)
            cols = [panel_pos[genes[k]] for k in in_panel]
            out[:, in_panel] = P[:, cols]
        if extra:
            if reference is None or gene_embeddings is None:
                raise ValueError("predicting genes outside the fitted target set needs reference= and gene_embeddings=")
            out[:, extra] = self._predict_extra([genes[k] for k in extra], cells, reference, gene_embeddings,
                                                batch_size, gene_chunk)
        if not as_anndata:
            return out, genes
        import anndata as ad
        import pandas as pd
        A = ad.AnnData(X=out, obs=pd.DataFrame(index=prep.obs_names[cells].astype(str)),
                       var=pd.DataFrame(index=pd.Index(genes, name="gene")))
        A.obsm["spatial"] = prep.xy[cells]
        split_all = np.full(prep.n_cells, "other", object)
        for nm, ix in (("train", prep.tr), ("val", prep.va), ("test", prep.te)):
            split_all[ix] = nm
        A.obs["split"] = split_all[cells]
        A.var["is_fit_target"] = [g in tg for g in genes]
        A.uns["resonate"] = dict(lambda_=float(net.a_scale), note="log1p of counts normalised to the median measured-gene "
                                 "library; see docs/inputs-outputs")
        return A

    # ------------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _predict_extra(self, genes, cells, reference, gene_embeddings, bs, chunk, reference_layer=None):
        prep, net, dev, cfg = self.prepared, self.net, self.device, self.cfg
        emb = load_gene_embeddings(gene_embeddings)
        if not np.array_equal(np.array(reference.obs_names.astype(str)), prep.ref_obs_names):
            raise ValueError("reference cells differ from those used in prepare() (obs_names / order)")
        rnames = {str(g): i for i, g in enumerate(reference.var_names)}
        bad = [g for g in genes if g not in rnames]
        if bad:
            raise ValueError(f"{len(bad)} genes are not in the reference, e.g. {bad[:5]}")
        noemb = [g for g in genes if not emb.has(g)]
        if noemb:
            raise ValueError(f"{len(noemb)} genes have no embedding, e.g. {noemb[:5]}")
        T, G = net.T, net.G
        vis = np.asarray(prep.vis)
        codes = prep.ref_codes.astype(np.int64)
        tr = prep.ref_tr
        REFt_vis = prep.REF[tr][:, vis]
        ct = codes[tr]
        names = list(prep.type_names)
        n_t = np.bincount(ct, minlength=T)
        Ew = REFt_vis.copy()
        for t in range(T):
            ix = np.where(ct == t)[0]
            if len(ix):
                Ew[ix] -= Ew[ix].mean(0, keepdims=True)
        V_pan = prep.V.astype(np.float64)
        Vpn = V_pan / (np.linalg.norm(V_pan, axis=1, keepdims=True) + 1e-8)
        C = reference.X if not reference_layer else reference.layers[reference_layer]
        C = sp.csr_matrix(C) if not sp.issparse(C) else C.tocsr()
        plib = prep.panel_library
        a_, b_ = self.summary["stage2"]["lookup_a"], self.summary["stage2"]["lookup_b"]
        res = np.zeros((len(cells), len(genes)), np.float32)
        full = np.zeros(G, bool)
        full[prep.vis] = True
        sf = np.broadcast_to(full, (bs, G))
        net.eval()
        for s in range(0, len(genes), chunk):
            ge = genes[s:s + chunk]
            K = len(ge)
            cols = [rnames[g] for g in ge]
            Ye = sp.diags(np.where(plib > 0, 1e4 / np.where(plib > 0, plib, 1.0), 0.0)) @ C[:, cols].astype(np.float64)
            Ye = Ye.tocsr()
            Ye.data = np.log1p(Ye.data)
            REF_e = Ye.toarray().astype(np.float32)
            Re_t = REF_e[tr]
            Ee = Re_t.copy()
            for t in range(T):
                ix = np.where(ct == t)[0]
                if len(ix):
                    Ee[ix] -= Ee[ix].mean(0, keepdims=True)
            A_W = cross_corr(Ee, Ew)
            A_ref = np.zeros((K, G), np.float32)
            A_ref[:, vis] = cross_corr(Re_t, REFt_vis)
            A_te = np.zeros((T, K, G), np.float32)
            for t in range(T):
                if names[t] == "small" or n_t[t] < self.cfg.min_type_cells:
                    R = A_W
                else:
                    R = (n_t[t] * cross_corr(Re_t[ct == t], REFt_vis[ct == t]) + self.cfg.shrink_n0 * A_W) / (n_t[t] + self.cfg.shrink_n0)
                A_te[t][:, vis] = R
            V_e = emb.lookup(ge).astype(np.float32)
            Ve64 = V_e.astype(np.float64)
            S_e = ((Ve64 / (np.linalg.norm(Ve64, axis=1, keepdims=True) + 1e-8)) @ Vpn.T).astype(np.float32)
            S_e[:, prep.oov] = 0.0
            N_idx, N_ok = negative_sets(A_ref, vis, self.cfg.n_neg_partners, self_idx=-np.ones(K, int))
            P_idx, P_ok = positive_sets(A_te, S_e, vis, N_idx, N_ok, self.cfg.n_pos_partners, self_idx=-np.ones(K, int))
            pair_idx = np.concatenate([P_idx, np.broadcast_to(N_idx[None], (T, K, N_idx.shape[1]))], 2)
            pair_ok = np.concatenate([P_ok, np.broadcast_to(N_ok[None], (T, K, N_ok.shape[1]))], 2)
            A_pair = np.take_along_axis(A_te, pair_idx, axis=2) * pair_ok
            t_ = lambda a, dt: torch.tensor(a, dtype=dt, device=dev)
            blk = TargetBlock(V=t_(V_e, torch.float32), S=t_(S_e, torch.float32), pair_idx=t_(pair_idx, torch.long),
                              pair_ok=t_(pair_ok, torch.bool), A_pair=t_(A_pair, torch.float32))
            L_e = (a_ * neighbor_mean(prep.nbr, REF_e) + b_).astype(np.float32)
            mu_e = net.mu_head(blk.V)
            W = net.readout_weights(net.gene_emb(), blk)
            for i in range(0, len(cells), bs):
                cb = cells[i:i + bs]
                m = np.broadcast_to(full, (len(cb), G)).copy()
                x = net.forward_extra(to_dev(cb, dev, np.int64), to_dev(m, dev), blk, to_dev(L_e[cb], dev), mu_e, W=W)
                res[i:i + len(cb), s:s + K] = x.cpu().numpy()
        return res

    # ------------------------------------------------------------------------------------------------------------------
    def save(self, path):
        """Write the trained weights, configuration and training summary (not the prepared inputs: ``Prepared.save``)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path / "model.pt")
        json.dump(dict(config=self.cfg.to_dict(), summary=self.summary, a_scale=float(self.net.a_scale)),
                  open(path / "model.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))

    @classmethod
    def load(cls, path, prepared: Prepared, device=None):
        """Re-create a fitted model from ``save`` output and the matching prepared inputs."""
        path = Path(path)
        meta = json.load(open(path / "model.json"))
        cfg = Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in meta["config"].items()})
        dev = device or cfg.device or default_device()
        _setup_torch(cfg, dev)
        parts = partner_sets(prepared, cfg)
        net = build_net(prepared, cfg, parts, dev, cfg.seed)
        net.load_state_dict(torch.load(path / "model.pt", map_location=dev, weights_only=True), strict=True)
        st = meta["summary"]["stage2"]
        a, b = st["lookup_a"], st["lookup_b"]
        L_te = (a * np.asarray(prepared.r_ref, np.float32) + b).astype(np.float32)
        net.set_data(stage=2, X=prepared.X, X_in=prepared.X_input, votes=prepared.votes_sp, pi=prepared.pi,
                     L_train=L_te, L_test=L_te, NB=prepared.NB, labels=prepared.labels_sp, device=dev)
        set_value_stats(net, prepared.X, prepared.tr, prepared.vis)     # feature standardisation (not in the checkpoint)
        net.mu_frozen = True
        net.a_scale = meta["a_scale"]
        return cls(net, prepared, cfg, meta["summary"], {}, dev)


def fit(prepared: Prepared, config: Config | None = None, *, max_train_cells=0, verbose=True) -> Resonate:
    """Two-stage training on prepared inputs: reference pre-training, then spatial fine-tuning.

    ``config`` should be the one used in :func:`prepare` (defaults reproduce the deployed model). ``max_train_cells``
    subsamples training cells and is meant for quick tests only."""
    cfg = config or Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in prepared.meta["config"].items()})
    device = cfg.device or default_device()
    _setup_torch(cfg, device)
    log = (lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)) if verbose else (lambda *a: None)
    parts = partner_sets(prepared, cfg)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    state1, s1, h1 = stage1(prepared, cfg, device, parts, max_train_cells, log)
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    net, s2, h2 = stage2(prepared, cfg, device, parts, state1, max_train_cells, log)
    return Resonate(net, prepared, cfg, dict(stage1=s1, stage2=s2), dict(stage1=h1, stage2=h2), device)


def predict(model: Resonate, genes=None, **kw):
    """Functional form of :meth:`Resonate.predict`."""
    return model.predict(genes, **kw)
