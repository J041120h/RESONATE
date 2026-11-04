"""The RESONATE model: x_hat(c, g) = M(c, g) + R(c, g) + lambda * A(c, g)   (Methods, equation 5).

M  mean term         1/2 mu(g) + 1/2 (beta1 * lookup + beta0); mu = MLP(embedding), lookup = mean over the 100 nearest
                     reference cells (the affine calibration is folded into the lookup tensors handed to ``set_data``).
R  correlation term  sum_t p_c(t) [ sum_{g' in P+_t(g)} w+ (x_g' - mu_g') - alpha * sum_{g' in P-(g)} w- (x_g' - mu_g') ]
A  attention term    see ``attention.py``; lambda = ``a_scale``.

Tensor layout (G = panel size = measured genes + target genes, T = reference cell types):
  * ``pair_idx`` (T, G, KP + KN): up to KP positive partners per type followed by KN negative partners (shared by types);
  * one matmul per step computes all four partner sums for every type (the readout weights are cell independent).
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

from .attention import AttentionTerm

N_FOLDS = 20
DEPLOY_SLOT = N_FOLDS          # slot 20 = all measured genes (used for target genes and at prediction time)
N_SLOTS = N_FOLDS + 1
K_NN = 100


class MLP(nn.Module):
    def __init__(self, in_d, hidden, out_d, dropout=0.1):
        super().__init__()
        layers, prev = [], in_d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.GELU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, out_d))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GeneEmbedding(nn.Module):
    """Fixed foundation-model gene embeddings; genes without an embedding share one learnable vector."""

    def __init__(self, V: np.ndarray, oov_mask: np.ndarray):
        super().__init__()
        self.register_buffer("V_invocab", torch.tensor(V, dtype=torch.float32))
        self.register_buffer("oov_mask", torch.tensor(oov_mask, dtype=torch.bool))
        self.has_oov = bool(oov_mask.any())
        if self.has_oov:
            self.oov_emb = nn.Parameter(self.V_invocab[~self.oov_mask].mean(dim=0).clone())
        else:
            self.register_parameter("oov_emb", None)

    def forward(self):
        if not self.has_oov:
            return self.V_invocab
        V = self.V_invocab.clone()
        V[self.oov_mask] = self.oov_emb
        return V


class MuHead(nn.Module):
    """mu(g) = MLP_mu(e_g): one gene-mean network, no cell-type input."""

    def __init__(self, emb_dim, hidden=(96, 96, 96)):
        super().__init__()
        self.mlp = MLP(emb_dim, list(hidden), 1, dropout=0.0)

    def forward(self, V):
        return self.mlp(V).squeeze(-1)


# ----------------------------------------------------------------------------------------------------------------------
# partner sets
# ----------------------------------------------------------------------------------------------------------------------
def negative_sets(A_ref, measured_idx, n_neg=8, self_idx=None):
    """Up to ``n_neg`` measured genes with the most negative reference correlation to each target row of ``A_ref``
    (K, G). ``self_idx[k]`` is the panel column of row k itself (excluded), or -1; default: row k is gene k."""
    A = np.nan_to_num(np.asarray(A_ref, np.float64), nan=0.0)
    vis = np.sort(np.asarray(measured_idx, np.int64))
    K = A.shape[0]
    self_idx = np.arange(K) if self_idx is None else np.asarray(self_idx)
    idx = np.zeros((K, n_neg), np.int64)
    ok = np.zeros((K, n_neg), bool)
    for g in range(K):
        cand = vis[vis != self_idx[g]]
        a = A[g, cand]
        neg = a < 0
        c, v = cand[neg], a[neg]
        order = np.lexsort((c, v))
        k = min(n_neg, len(order))
        idx[g, :k] = c[order[:k]]
        ok[g, :k] = True
    return idx, ok


def positive_sets(A_t, S_FM, measured_idx, neg_idx, neg_ok, k_pos=128, self_idx=None):
    """Per type, up to ``k_pos`` measured genes with the strongest positive type-specific correlation (ties broken by
    embedding cosine, then index); disjoint from the gene itself and from its negative partners. ``A_t`` is (T, K, G)
    and ``S_FM`` (K, G) for K target rows."""
    T, K, _ = A_t.shape
    S = np.nan_to_num(np.asarray(S_FM, np.float64), nan=0.0)
    vis = np.sort(np.asarray(measured_idx, np.int64))
    self_idx = np.arange(K) if self_idx is None else np.asarray(self_idx)
    idx = np.zeros((T, K, k_pos), np.int64)
    ok = np.zeros((T, K, k_pos), bool)
    for g in range(K):
        excl = set(neg_idx[g][neg_ok[g]].tolist()) | {int(self_idx[g])}
        cand = np.array([s for s in vis if s not in excl], np.int64)
        for t in range(T):
            a = np.nan_to_num(np.asarray(A_t[t, g, cand], np.float64), nan=0.0)
            pos = a > 0
            c, v = cand[pos], a[pos]
            order = np.lexsort((c, -S[g, c], -v))
            k = min(k_pos, len(order))
            idx[t, g, :k] = c[order[:k]]
            ok[t, g, :k] = True
    return idx, ok


# ----------------------------------------------------------------------------------------------------------------------
class TargetBlock:
    """Gene-specific inputs of K target rows: embeddings V (K, D), cosine to the G panel genes S (K, G), partner indices
    / validity (T, K, KP + KN) and the type-specific correlations A_pair (T, K, KP + KN) of the partner pairs."""

    def __init__(self, V, S, pair_idx, pair_ok, A_pair):
        self.V, self.S, self.pair_idx, self.pair_ok, self.A_pair = V, S, pair_idx, pair_ok, A_pair


class ResonateNet(nn.Module):
    def __init__(self, *, V, oov_mask, S_FM, A_t, P_idx, P_ok, N_idx, N_ok, gene_slot, target_idx, type_dist, seed=0,
                 dropout=0.1, mu_hidden=(96, 96, 96), corr_hidden=32, n_nb=15, adapter_seed=20261018,
                 attn_impl="sdpa"):
        super().__init__()
        G, D = V.shape
        T = A_t.shape[0]
        self.G, self.T, self.D = G, T, D
        assert np.asarray(type_dist).shape == (T, T)
        torch.manual_seed(seed)
        self.register_buffer("S_FM", torch.tensor(np.nan_to_num(S_FM, nan=0.0), dtype=torch.float32))
        KP, KN = P_idx.shape[2], N_idx.shape[1]
        pair_idx = np.concatenate([P_idx, np.broadcast_to(N_idx[None], (T, G, KN))], 2)
        pair_ok = np.concatenate([P_ok, np.broadcast_to(N_ok[None], (T, G, KN))], 2)
        self.KP, self.KN = int(KP), int(KN)
        self.register_buffer("pair_idx", torch.tensor(pair_idx, dtype=torch.long))
        self.register_buffer("pair_ok", torch.tensor(pair_ok, dtype=torch.bool))
        A_pair = np.take_along_axis(np.nan_to_num(A_t, nan=0.0), pair_idx, axis=2) * pair_ok
        self.register_buffer("A_pair", torch.tensor(A_pair, dtype=torch.float32))
        self.register_buffer("gene_slot", torch.tensor(np.asarray(gene_slot, np.int64)))
        held = np.zeros(G, bool)
        held[np.asarray(target_idx, np.int64)] = True
        self.register_buffer("target_mask", torch.tensor(held))
        # data, not a parameter: always taken from the prepared inputs, never from a checkpoint
        self.register_buffer("type_dist", torch.tensor(np.asarray(type_dist, np.float32)), persistent=False)
        # parameters (creation order fixes the initial weights for a given seed)
        self.gene_emb = GeneEmbedding(np.asarray(V, np.float32), np.asarray(oov_mask, bool))
        self.mu_head = MuHead(D, mu_hidden)
        self.corr_mlp = MLP(2 * D + 2, [corr_hidden], 1, dropout=dropout)
        nn.init.zeros_(self.corr_mlp.net[-1].weight)
        nn.init.zeros_(self.corr_mlp.net[-1].bias)
        g = torch.Generator().manual_seed(adapter_seed)
        bound = 1.0 / math.sqrt(2 * D + 2)
        self.adapter_a = nn.Parameter((torch.rand(T, 2 * D + 2, generator=g) * 2 - 1) * bound)   # rank-one type adapter
        self.adapter_u = nn.Parameter(torch.zeros(T, corr_hidden))
        self.alpha_neg = nn.Parameter(torch.zeros(()))
        devs = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devs):
            torch.manual_seed(10_000 + 97 * 9 + int(seed))
            self.attn = AttentionTerm(G, fm_dim=D, n_nb=n_nb, dropout=dropout, dropout_seed=int(seed))
        self.attn.set_impl(attn_impl)
        # mu is frozen after each stage's warm-up: its level is not identified otherwise (A can cancel a uniform shift)
        self.register_buffer("mu_fixed", torch.zeros(G))
        self.mu_frozen = False
        self.stage = None
        self.a_off = False
        self._ro_cache = None
        self.attn_bf16 = False        # bf16 autocast inside the attention term (CUDA only)
        self.a_scale = 1.0            # lambda
        self.data_device_req = None

    # ------------------------------------------------------------------------------------------------------------------
    def set_data(self, *, stage, X, X_in, votes, pi, L_train, L_test=None, NB=None, labels=None, device="cpu"):
        """X (N, G) log-normalised values (targets; attention input); X_in (N, G) correlation-term input (smoothed
        spatial / unsmoothed reference); votes (N, 21, T) uint8; pi (T,); L_train / L_test (N, G) calibrated lookups;
        NB (N, n_nb) spatial-neighbour rows and labels (N,) hard cell types (stage 2 only). Target columns of X_in are
        zeroed here."""
        assert stage in (1, 2)
        assert (NB is None) == (labels is None) == (stage == 1)
        self.stage = stage
        held = self.target_mask.cpu().numpy()
        dd = self.data_device = self.data_device_req or device
        f32 = lambda a: torch.tensor(np.ascontiguousarray(a, np.float32), device=dd)
        f32h = lambda a: torch.tensor(np.ascontiguousarray(a, np.float32), device="cpu")
        Xi = np.array(X_in, np.float32, copy=True)
        Xi[:, held] = 0.0
        self.register_buffer("X", f32(X), persistent=False)
        self.register_buffer("X_in", f32(Xi), persistent=False)
        del Xi
        v = np.asarray(votes)
        assert v.dtype == np.uint8 and v.shape[1:] == (N_SLOTS, self.T) and (v.sum(-1) == K_NN).all()
        self.register_buffer("votes", torch.tensor(v, device=dd), persistent=False)
        self.register_buffer("pi", torch.tensor(np.asarray(pi, np.float32), device=device), persistent=False)
        self.register_buffer("L_train", f32(np.nan_to_num(L_train, nan=0.0)), persistent=False)
        self.L_test = f32h(np.nan_to_num(L_test if L_test is not None else L_train, nan=0.0))
        self.NB = torch.tensor(np.asarray(NB, np.int64), device=dd) if NB is not None else None
        self.labels = torch.tensor(np.asarray(labels, np.int64), device=dd) if labels is not None else None
        self.compute_device = torch.device(device)

    def _rows(self, buf, idx):
        if buf.device == idx.device:
            return buf[idx]
        return buf[idx.to(buf.device)].to(idx.device, non_blocking=True)

    def X_a_rows(self, idx):
        """rows of X with the target columns zeroed (what the attention term reads)."""
        return self._rows(self.X, idx).masked_fill(self.target_mask, 0.0)

    # ------------------------------------------------------------------------------------------------------------------
    def mu0(self):
        return self.mu_fixed if self.mu_frozen else self.mu_head(self.gene_emb())

    @torch.no_grad()
    def freeze_mu(self):
        self.mu_frozen = False
        self.mu_fixed.copy_(self.mu_head(self.gene_emb()))
        self.mu_frozen = True
        for p in self.mu_head.parameters():
            p.requires_grad_(False)

    def panel_block(self, V):
        """The target block of the fitted panel: every panel gene is a target row."""
        return TargetBlock(V=V, S=self.S_FM, pair_idx=self.pair_idx, pair_ok=self.pair_ok, A_pair=self.A_pair)

    def _pair_scores(self, Vsrc, blk):
        """pair score eta_t(g, g') = MLP_w([e_g; e_g'; cos; rho_t]) + rank-one type adapter on the (T, K, KP + KN)
        pairs of the K target rows in ``blk``; sources g' index the panel embeddings ``Vsrc`` (G, D)."""
        lin1 = self.corr_mlp.net[0]
        D = self.D
        Wt, b1 = lin1.weight, lin1.bias
        W_g, W_s, W_c, W_A = Wt[:, :D], Wt[:, D:2 * D], Wt[:, 2 * D], Wt[:, 2 * D + 1]
        T, K, _ = blk.pair_idx.shape
        idx = blk.pair_idx
        Pg = blk.V @ W_g.t()
        Qs = Vsrc @ W_s.t()
        rows = torch.arange(K, device=Vsrc.device).view(1, K, 1)
        cos = blk.S[rows, idx]
        h1 = Pg.view(1, K, 1, -1) + Qs[idx] + cos.unsqueeze(-1) * W_c + blk.A_pair.unsqueeze(-1) * W_A + b1
        a, u = self.adapter_a, self.adapter_u
        al = blk.V @ a[:, :D].t()
        be = Vsrc @ a[:, D:2 * D].t()
        tt = torch.arange(T, device=Vsrc.device).view(T, 1, 1)
        zz = (al.t().unsqueeze(2) + be.t()[tt, idx] + a[:, 2 * D].view(T, 1, 1) * cos
              + a[:, 2 * D + 1].view(T, 1, 1) * blk.A_pair)
        h1 = h1 + zz.unsqueeze(-1) * u.view(T, 1, 1, -1)
        for layer in self.corr_mlp.net[1:]:
            h1 = layer(h1)
        return h1.squeeze(-1)

    def _dense_exp(self, logits, ok, idx, n_src):
        Lm = logits.masked_fill(~ok, float("-inf"))
        rm = Lm.max(-1, keepdim=True).values.detach()
        rm = torch.where(torch.isfinite(rm), rm, torch.zeros_like(rm))
        E = torch.exp(Lm - rm) * ok
        return E.new_zeros(logits.shape[0], logits.shape[1], n_src).scatter_add(2, idx, E)

    def readout_weights(self, Vsrc=None, blk=None):
        """W (2T, K, G) = [E+ ; E-], the un-normalised positive / negative softmax numerators of the K target rows."""
        if blk is None and self._ro_cache is not None:
            return self._ro_cache
        Vsrc = self.gene_emb() if Vsrc is None else Vsrc
        blk = self.panel_block(Vsrc) if blk is None else blk
        S = self._pair_scores(Vsrc, blk)
        KP = self.KP
        Ep = self._dense_exp(S[..., :KP], blk.pair_ok[..., :KP], blk.pair_idx[..., :KP], Vsrc.shape[0])
        En = self._dense_exp(-S[..., KP:], blk.pair_ok[..., KP:], blk.pair_idx[..., KP:], Vsrc.shape[0])
        return torch.cat([Ep, En], 0)

    @contextmanager
    def cache_readout(self):
        """eval only: compute the (cell independent) readout weights once for many batches."""
        assert not self.training and not torch.is_grad_enabled()
        self._ro_cache = None
        self._ro_cache = self.readout_weights()
        try:
            yield
        finally:
            self._ro_cache = None

    def _correlation_sums(self, y, m, W):
        """(B, 2, T, K): positive / negative weighted means of the partner deviations for every type."""
        B = y.shape[0]
        T, K = W.shape[0] // 2, W.shape[1]
        S = torch.cat([y, m], 0) @ W.reshape(2 * T * K, W.shape[2]).t()
        S = S.view(2, B, 2, T, K)
        num, den = S[0], S[1]
        return torch.where(den > 0, num / torch.where(den > 0, den, torch.ones_like(den)), torch.zeros_like(num))

    # ------------------------------------------------------------------------------------------------------------------
    def forward(self, batch_idx, input_mask, lookup="train", aux_V=None, return_parts=False):
        assert self.stage is not None, "set_data() first"
        assert lookup in ("train", "test")
        V = self.gene_emb()
        m = input_mask.to(V.dtype)
        B, G, T = batch_idx.shape[0], self.G, self.T
        mu = self.mu_fixed if self.mu_frozen else self.mu_head(V)
        y = (self._rows(self.X_in, batch_idx) - mu.unsqueeze(0)) * m
        safe = self._correlation_sums(y, m, self.readout_weights(V))
        C_t = safe[:, 0] - self.alpha_neg * safe[:, 1]
        p = (self._rows(self.votes, batch_idx).to(V.dtype) + self.pi.to(V.device)) / float(K_NN + 1)
        p_g = p[:, self.gene_slot, :]
        R = torch.einsum("bgt,btg->bg", p_g, C_t)
        L = self._rows(self.L_train if lookup == "train" else self.L_test, batch_idx)
        M = 0.5 * mu.unsqueeze(0) + 0.5 * L
        x_hat = M + R
        A = A_aux = None
        if not self.a_off:
            if self.NB is not None:
                nb_rows = self._rows(self.NB, batch_idx)
                lab_c = self._rows(self.labels, batch_idx)
                lab_n = self._rows(self.labels, nb_rows)
                nb_dist = self.type_dist[lab_c.unsqueeze(1), lab_n]
                if self.X.device == batch_idx.device:
                    nb = lambda order: self.X[nb_rows.unsqueeze(-1), order.unsqueeze(1)]
                else:
                    Xn = self._rows(self.X, nb_rows.reshape(-1)).view(*nb_rows.shape, G)
                    nb = lambda order: torch.gather(Xn, 2, order.unsqueeze(1).expand(-1, nb_rows.shape[1], -1))
            else:
                nb = nb_dist = None
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.attn_bf16 and batch_idx.is_cuda):
                A, A_aux = self.attn(V.detach(), self.X_a_rows(batch_idx), input_mask, nb, nb_dist, aux_V)
            A = A.float()
            A_aux = A_aux.float() if A_aux is not None else None
            x_hat = x_hat + (A if self.a_scale == 1.0 else self.a_scale * A)
        if return_parts:
            return x_hat, dict(M=M, R=R, A=A, A_aux=A_aux, mu=mu, y=y, p=p)
        return x_hat

    @torch.no_grad()
    def forward_extra(self, batch_idx, input_mask, blk, L_extra, mu_extra, W=None):
        """Prediction for target genes outside the fitted panel (``blk`` holds their rows). The same trained
        networks are applied; only the gene-specific inputs (embedding, partners, correlations, lookup) differ."""
        assert self.stage == 2
        V = self.gene_emb()
        m = input_mask.to(V.dtype)
        T = self.T
        y = (self._rows(self.X_in, batch_idx) - self.mu_fixed.unsqueeze(0)) * m
        W = self.readout_weights(V, blk) if W is None else W
        safe = self._correlation_sums(y, m, W)
        C_t = safe[:, 0] - self.alpha_neg * safe[:, 1]                                  # (B, T, K)
        p = (self._rows(self.votes, batch_idx)[:, DEPLOY_SLOT, :].to(V.dtype) + self.pi.to(V.device)) / float(K_NN + 1)
        R = torch.einsum("bt,btk->bk", p, C_t)
        x_hat = 0.5 * mu_extra.unsqueeze(0) + 0.5 * L_extra + R
        if not self.a_off:
            nb_rows = self._rows(self.NB, batch_idx)
            lab_c = self._rows(self.labels, batch_idx)
            lab_n = self._rows(self.labels, nb_rows)
            nb_dist = self.type_dist[lab_c.unsqueeze(1), lab_n]
            nb = lambda order: self.X[nb_rows.unsqueeze(-1), order.unsqueeze(1)]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.attn_bf16 and batch_idx.is_cuda):
                PV, mem, valid = self.attn.encode(V.detach(), self.X_a_rows(batch_idx), input_mask, nb, nb_dist)
                A = self.attn.decode(self.attn.P(blk.V.detach()), mem, valid)
            x_hat = x_hat + self.a_scale * A.float()
        return x_hat

    def post_step(self):
        with torch.no_grad():
            self.alpha_neg.clamp_(0.0, 1.0)
