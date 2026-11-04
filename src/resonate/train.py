"""Two-stage training: reference pre-training, then spatial fine-tuning (Methods, "Training and prediction")."""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import torch

from .config import Config
from .metrics import per_gene_pearson
from .model.model import ResonateNet, negative_sets, positive_sets


# ---------------------------------------------------------------------------------------------------------------------
# masks and loss
# ---------------------------------------------------------------------------------------------------------------------
def make_train_masks(B, G, pool, targets, rng, frac_cell=0.20, frac_batch=0.05):
    """Per-batch masks over the stage's target genes ``pool``.

    ``batch`` mask: the same ceil(5 %) of the pool hidden in every cell. ``cell`` mask: ceil(20 %) drawn independently
    in each cell (disjoint from the batch mask). Genes in ``targets`` (never measured) are never inputs.
    Returns input_mask, cell_mask, batch_mask, each (B, G) bool."""
    pool = np.asarray(pool)
    n = len(pool)
    n_B = max(1, int(np.ceil(frac_batch * n))) if frac_batch > 0 else 0
    B_pos = np.zeros((B, G), bool)
    if n_B:
        B_pos[:, pool[rng.choice(n, size=n_B, replace=False)]] = True
    n_A = max(1, int(np.ceil(frac_cell * n))) if frac_cell > 0 else 0
    A_pos = np.zeros((B, G), bool)
    if n_A:
        scores = rng.random((B, n))
        a_local = np.argpartition(scores, n_A - 1, axis=1)[:, :n_A]
        A_pos[np.repeat(np.arange(B), n_A), pool[a_local].ravel()] = True
        A_pos &= ~B_pos
    inp = np.zeros((B, G), bool)
    inp[:, pool] = True
    inp &= ~A_pos & ~B_pos
    inp[:, targets] = False
    return inp, A_pos, B_pos


def tier_loss(pred, target, A_t, B_t, a_any, b_any, w_A=0.6, w_B=0.4):
    """0.6 * MSE over the per-cell masked pairs + 0.4 * MSE over the per-batch masked pairs."""
    se = (pred - target) ** 2
    parts = []
    if a_any:
        parts.append(w_A * se[A_t].mean())
    if b_any:
        parts.append(w_B * se[B_t].mean())
    if not parts:
        return torch.zeros((), device=pred.device, requires_grad=True)
    return sum(parts)


def fit_global_affine(r_ref, X, rows, cols):
    """Pooled least squares of X on the lookup over (training cells x measured genes): the calibration (beta1, beta0)."""
    r = r_ref[np.ix_(rows, cols)].ravel()
    x = X[np.ix_(rows, cols)].ravel()
    m = np.isfinite(r) & np.isfinite(x)
    A = np.vstack([r[m], np.ones(int(m.sum()), np.float64)]).T
    a, b = np.linalg.lstsq(A, x[m].astype(np.float64), rcond=None)[0]
    return float(a), float(b)


def to_dev(a, device, dtype=None):
    return torch.from_numpy(np.ascontiguousarray(a if dtype is None else np.asarray(a, dtype))).to(device)


# ---------------------------------------------------------------------------------------------------------------------
# helpers shared by both stages
# ---------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def set_value_stats(model, X, rows, genes):
    """Per-gene mean / sd of the training cells (feature standardisation of the attention term)."""
    rows = np.sort(np.asarray(rows))
    g = np.asarray(genes)
    s1 = np.zeros(len(g))
    s2 = np.zeros(len(g))
    n = 0
    for i in range(0, len(rows), 65536):
        sub = np.asarray(X[rows[i:i + 65536]][:, g], np.float64)
        s1 += sub.sum(0)
        s2 += (sub * sub).sum(0)
        n += len(sub)
    mu = s1 / n
    sd = np.sqrt(np.maximum(s2 / n - mu ** 2, 0.0))
    mean = np.zeros(model.G, np.float32)
    sdev = np.ones(model.G, np.float32)
    mean[g] = mu
    sdev[g] = sd
    model.attn.set_value_stats(mean, sdev)


def mean_warmup(model, targets_mean, genes, device, steps=200, lr=1e-3):
    """mu(g) -> per-gene means: full-batch Adam on the mean network and the shared out-of-vocabulary vector."""
    params = list(model.mu_head.parameters()) + ([model.gene_emb.oov_emb] if model.gene_emb.has_oov else [])
    opt = torch.optim.Adam(params, lr=lr)
    g = torch.as_tensor(np.asarray(genes), device=device)
    tgt = torch.as_tensor(np.asarray(targets_mean, np.float32), device=device)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model.mu0()[g], tgt)
        loss.backward()
        opt.step()
    return float(loss.detach())


@torch.no_grad()
def predict_cells(model, cells, in_mask_fn, device, lookup, bs=512):
    """Run the model over ``cells`` (row indices) with the input masks given by ``in_mask_fn(start, n)``."""
    model.eval()
    out = []
    held = model.target_mask.cpu().numpy()
    with model.cache_readout():
        for i in range(0, len(cells), bs):
            cb = np.asarray(cells[i:i + bs])
            m = in_mask_fn(i, len(cb))
            assert not m[:, held].any(), "a target gene is in an input mask"
            out.append(model(to_dev(cb, device, np.int64), to_dev(m, device), lookup=lookup).cpu().numpy())
    return np.concatenate(out, 0) if out else np.zeros((0, model.G), np.float32)


class Aux:
    """Auxiliary pre-training targets: 256 of the 2,000 reference genes outside the panel are drawn for every batch;
    only the attention term is updated by their loss (target mean and an unweighted partner average stand in for the
    mean and correlation terms)."""

    def __init__(self, prep, vis, device, n_per_batch=256, seed=300):
        Cm = sp.csr_matrix((prep.aux_counts_data, prep.aux_counts_indices, prep.aux_counts_indptr),
                           shape=tuple(prep.aux_counts_shape))
        lib = np.asarray(prep.aux_panel_library, np.float64)
        valid = lib > 0
        scale = np.where(valid, prep.meta["aux_target_sum"] / np.where(valid, lib, 1.0), 0.0).astype(np.float32)
        Y = (sp.diags(scale) @ Cm.astype(np.float32)).tocsr()
        Y.data = np.log1p(Y.data)
        tr = np.sort(np.asarray(prep.ref_tr))
        trv = tr[valid[tr]]
        m = np.asarray(Y[trv].sum(0), np.float64).ravel() / max(len(trv), 1)
        self.Y = torch.from_numpy(Y.toarray()).to(device)
        self.w = torch.from_numpy(valid.astype(np.float32)).to(device)
        self.m = torch.from_numpy(m.astype(np.float32)).to(device)
        self.V = torch.from_numpy(np.asarray(prep.aux_V, np.float32)).to(device)
        A_aux = np.nan_to_num(np.asarray(prep.aux_A, np.float64), nan=0.0)
        nrm = lambda M: M / (np.linalg.norm(M.astype(np.float64), axis=1, keepdims=True) + 1e-8)
        S_aux = (nrm(prep.aux_V) @ nrm(prep.V.astype(np.float64)).T).astype(np.float32)
        S_aux[:, prep.oov] = 0.0
        Pn, G = A_aux.shape
        pm = np.zeros((Pn, G), bool)
        visv = np.sort(np.asarray(vis))
        for a in range(Pn):                                  # up to 128 positive partners by reference correlation
            v = A_aux[a, visv]
            c, vv = visv[v > 0], v[v > 0]
            o = np.lexsort((c, -S_aux[a, c], -vv))[:128]
            pm[a, c[o]] = True
        self.partner = torch.from_numpy(pm.astype(np.float32)).to(device)
        self.P, self.n = Pn, int(min(n_per_batch, Pn))
        self.rng = np.random.default_rng(seed)

    def sample(self):
        return torch.from_numpy(np.sort(self.rng.choice(self.P, self.n, replace=False))).to(self.Y.device)

    def loss(self, in_t, parts, idx_t, sel):
        with torch.no_grad():
            m = in_t.float()
            y = parts["y"].detach()
            Pm = self.partner[sel]
            num, den = y @ Pm.t(), m @ Pm.t()
            R_aux = torch.where(den > 0, num / den.clamp_min(1e-12), torch.zeros_like(num))
        pred = self.m[sel].unsqueeze(0) + R_aux + parts["A_aux"]
        tgt = self.Y[idx_t][:, sel]
        w = self.w[idx_t].unsqueeze(1)
        return (((pred - tgt) ** 2) * w).sum() / (w.sum() * pred.shape[1]).clamp_min(1.0)


def fit_loop(model, *, train_idx, pool, targets, val_fn, device, seed, max_epochs, patience, lr, batch_size, cfg,
             aux=None, always_in=None, name="stage", log=print):
    """Adam, per-epoch validation, early stopping; the best epoch's trainable parameters are restored."""
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=lr)
    rng = np.random.default_rng(seed)
    G = model.G
    best, best_ep, stale, best_state, hist = -np.inf, 0, 0, None, []
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        model.train()
        te = time.time()
        tr_sum = torch.zeros((), dtype=torch.float64, device=device)
        aux_sum = tr_sum.clone()
        n = 0
        ti = np.asarray(train_idx)[rng.permutation(len(train_idx))]
        for i in range(0, len(ti), batch_size):
            b = ti[i:i + batch_size]
            in_np, A_np, B_np = make_train_masks(len(b), G, pool, targets, rng, cfg.mask_frac_cell, cfg.mask_frac_batch)
            if always_in is not None:                    # pseudo-held genes: always inputs, never targets
                in_np[:, always_in] = True
            idx_t, in_t = to_dev(b, device, np.int64), to_dev(in_np, device)
            A_t, B_t = to_dev(A_np, device), to_dev(B_np, device)
            if aux is not None:
                sel = aux.sample()
                pred, parts = model(idx_t, in_t, lookup="train", aux_V=aux.V[sel], return_parts=True)
            else:
                pred = model(idx_t, in_t, lookup="train")
            loss = tier_loss(pred, model._rows(model.X, idx_t), A_t, B_t, bool(A_np.any()), bool(B_np.any()))
            total = loss
            if aux is not None:
                la = aux.loss(in_t, parts, idx_t, sel)
                total = loss + la
                aux_sum += la.detach().double() * len(b)
            opt.zero_grad()
            total.backward()
            opt.step()
            model.post_step()
            tr_sum += loss.detach().double() * len(b)
            n += len(b)
        score = val_fn()
        row = dict(stage=name, epoch=ep, train_loss=float(tr_sum / n), val_score=score, epoch_sec=time.time() - te,
                   alpha_neg=float(model.alpha_neg.detach()))
        if aux is not None:
            row["train_loss_aux"] = float(aux_sum / n)
        hist.append(row)
        log(f"  [{name}] epoch {ep:3d}  train {row['train_loss']:.5f}  val {score:.5f}  ({row['epoch_sec']:.0f}s)")
        if score > best + 1e-6:
            best, best_ep, stale = score, ep, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
        else:
            stale += 1
            if stale >= patience:
                log(f"  [{name}] early stop at epoch {ep}; best {best:.5f} at epoch {best_ep}")
                break
    with torch.no_grad():
        pd_ = dict(model.named_parameters())
        for k, v in best_state.items():
            pd_[k].copy_(v.to(pd_[k].device))
    return hist, best, best_ep, (time.time() - t0) / 60


def build_net(prep, cfg: Config, parts, device, seed):
    P_idx, P_ok, N_idx, N_ok = parts
    net = ResonateNet(V=prep.V, oov_mask=prep.oov, S_FM=prep.S_FM, A_t=prep.A_t, P_idx=P_idx, P_ok=P_ok, N_idx=N_idx,
                      N_ok=N_ok, gene_slot=prep.gene_slot, target_idx=prep.targets, type_dist=prep.type_dist, seed=seed,
                      dropout=cfg.dropout, mu_hidden=cfg.mu_hidden, corr_hidden=cfg.corr_hidden,
                      n_nb=cfg.n_spatial_nb, attn_impl=cfg.attn_impl).to(device)
    net.attn_bf16 = bool(cfg.attn_bf16)
    net.data_device_req = "cpu" if cfg.data_on_host else None
    return net


def partner_sets(prep, cfg):
    N_idx, N_ok = negative_sets(prep.A_ref_tr, prep.vis, cfg.n_neg_partners)
    P_idx, P_ok = positive_sets(prep.A_t, prep.S_FM, prep.vis, N_idx, N_ok, cfg.n_pos_partners)
    return P_idx, P_ok, N_idx, N_ok


# ---------------------------------------------------------------------------------------------------------------------
def stage1(prep, cfg, device, parts, max_train_cells=0, log=print):
    """Reference pre-training. Returns (state_dict, summary, history)."""
    REF = prep.REF
    G = REF.shape[1]
    targets, vis = np.asarray(prep.targets), np.asarray(prep.vis)
    pool = np.arange(G)                                         # every panel gene is a reconstruction target
    tr, va = prep.ref_tr, prep.ref_va
    if max_train_cells:
        tr = np.random.default_rng(7).permutation(tr)[:max_train_cells]
        va = va[:max(200, max_train_cells // 9)]
    model = build_net(prep, cfg, parts, device, cfg.seed)
    L_cv = np.asarray(prep.L_cv, np.float32)
    a1, b1 = fit_global_affine(L_cv, REF, tr, vis)
    L1 = np.where(np.isfinite(L_cv), a1 * L_cv + b1, 0.0).astype(np.float32)
    model.set_data(stage=1, X=REF, X_in=REF, votes=prep.votes_ref, pi=prep.pi, L_train=L1, L_test=L1, device=device)
    del L1
    set_value_stats(model, REF, tr, vis)
    wl = mean_warmup(model, np.asarray(REF[np.sort(tr)][:, pool], np.float64).mean(0), pool, device,
                     cfg.mu_warmup_steps, cfg.mu_warmup_lr)
    model.freeze_mu()
    log(f"[stage 1] {REF.shape[0]} reference cells ({len(tr)} train / {len(va)} validation); lookup affine "
        f"a={a1:.4f} b={b1:.4f}; mean warm-up loss {wl:.5f}")
    aux = Aux(prep, vis, device, cfg.aux_per_batch, seed=cfg.seed + 300)
    truth_va = np.asarray(REF[np.asarray(va)], np.float32)[:, targets]
    in_h = np.zeros((len(va), G), bool)
    in_h[:, pool] = True
    in_h[:, targets] = False

    def val_fn():
        P = predict_cells(model, va, lambda i, n: in_h[i:i + n], device, "train")
        return float(np.nanmean(per_gene_pearson(truth_va, P[:, targets])))
    hist, best, best_ep, mins = fit_loop(model, train_idx=tr, pool=pool, targets=targets, val_fn=val_fn, device=device,
                                         seed=cfg.seed + 100, max_epochs=cfg.max_epochs, patience=cfg.patience,
                                         lr=cfg.stage1_lr, batch_size=cfg.stage1_batch_size, cfg=cfg, aux=aux,
                                         name="stage1_reference", log=log)
    summ = dict(stage=1, best_ref_val_target_r=best, best_epoch=best_ep, n_epochs=len(hist), minutes=mins,
                lookup_a=a1, lookup_b=b1, alpha_neg=float(model.alpha_neg.detach()))
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, summ, hist


def pseudo_held(X, tr, vis, frac, seed):
    """``frac`` of the measured genes drawn among those detected in >= 10 % of training cells (stand-ins for unmeasured
    genes during fine-tuning). Uses training cells and measured genes only."""
    rows = np.sort(np.asarray(tr))
    rows = rows[np.linspace(0, len(rows) - 1, min(len(rows), 50000)).astype(int)]
    det = (np.asarray(X[rows][:, vis]) > 0).mean(0)
    pool = np.asarray(vis)[det >= 0.10]
    k = max(5, int(round(frac * len(vis))))
    return np.sort(np.random.default_rng(seed).choice(pool, min(k, len(pool)), replace=False))


def stage2(prep, cfg, device, parts, state1, max_train_cells=0, log=print):
    """Spatial fine-tuning from the pre-trained weights. Returns (net, summary, history)."""
    X, Xs, NB = prep.X, prep.X_input, prep.NB
    tr, va, vis, targets = prep.tr, prep.va, np.asarray(prep.vis), np.asarray(prep.targets)
    if max_train_cells:
        tr = np.random.default_rng(7).permutation(tr)[:max_train_cells]
        va = va[:max(200, max_train_cells // 9)]
    G = X.shape[1]
    model = build_net(prep, cfg, parts, device, cfg.seed)
    model.load_state_dict(state1, strict=True)
    r_cv = np.asarray(prep.r_ref_cv, np.float32)
    a, b = fit_global_affine(r_cv, X, tr, vis)
    L_tr = np.where(np.isfinite(r_cv), a * r_cv + b, 0.0).astype(np.float32)
    del r_cv
    assert np.all(L_tr[:, targets] == 0), "the gene-fold lookup touches a target gene"
    L_te = (a * np.asarray(prep.r_ref, np.float32) + b).astype(np.float32)
    model.set_data(stage=2, X=X, X_in=Xs, votes=prep.votes_sp, pi=prep.pi, L_train=L_tr, L_test=L_te, NB=NB,
                   labels=prep.labels_sp, device=device)
    del L_tr, L_te
    set_value_stats(model, X, tr, vis)
    wl = mean_warmup(model, np.asarray(X[np.sort(tr)][:, vis], np.float64).mean(0), vis, device,
                     cfg.mu_warmup_steps, cfg.mu_warmup_lr)
    model.freeze_mu()
    PH = pseudo_held(X, tr, vis, cfg.pseudo_held_fraction, seed=cfg.seed + 123)
    in_ph = np.zeros(G, bool)
    in_ph[vis] = True
    in_ph[PH] = False
    truth_ph = np.asarray(X[np.asarray(va)])[:, PH]

    def val_fn():
        P = predict_cells(model, va, lambda i, n: np.broadcast_to(in_ph, (n, G)).copy(), device, "train")
        return float(np.nanmean(per_gene_pearson(truth_ph, P[:, PH])))
    log(f"[stage 2] {len(tr)} train / {len(va)} validation cells; lookup affine a={a:.4f} b={b:.4f}; {len(PH)} of "
        f"{len(vis)} measured genes set aside to select the checkpoint and lambda")
    hist, best, best_ep, mins = fit_loop(model, train_idx=tr, pool=np.setdiff1d(vis, PH), targets=targets, val_fn=val_fn,
                                         device=device, seed=cfg.seed + 100, max_epochs=cfg.max_epochs,
                                         patience=cfg.patience, lr=cfg.stage2_lr, batch_size=cfg.stage2_batch_size,
                                         cfg=cfg, always_in=PH, name="stage2_spatial", log=log)
    grid = {}
    for lam in cfg.lambda_grid:
        model.a_scale = lam
        grid[lam] = val_fn()
    model.a_scale = max(grid, key=lambda k: (grid[k], k))
    log("[stage 2] lambda on the set-aside genes: " + ", ".join(f"{k:g}: {v:.4f}" for k, v in grid.items())
        + f" -> {model.a_scale:g}")
    summ = dict(stage=2, best_val_pseudo_held_r=best, best_epoch=best_ep, n_epochs=len(hist), minutes=mins, lookup_a=a,
                lookup_b=b, lambda_=float(model.a_scale), lambda_grid={str(k): v for k, v in grid.items()},
                alpha_neg=float(model.alpha_neg.detach()), pseudo_held=[int(g) for g in PH])
    return model, summ, hist
