"""Attention term A of the RESONATE model (Methods, "Attention term").

Tokens: every measured, unmasked gene g' of a cell becomes ``P e_g' + phi(f_cg')``. A two-layer pre-LN transformer
encoder (width 64, 4 heads, FFN 128, no positional encoding) lets the tokens interact; the target gene's projected
embedding is a query that cross-attends over the encoded tokens and a linear read-out returns one scalar per
(cell, target gene).

Numerics kept from the deployed model: dropout 0.1 on the residual branches drawn from module-private seeded
generators (the global RNG stream is never shifted), value clipping to [-10, 10], and the read-out layer is
zero-initialised (A == 0 at initialisation).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _DetSDPA(torch.autograd.Function):
    """Fused attention with a deterministic backward (the forward runs the fused kernel)."""

    @staticmethod
    def forward(ctx, q, k, v, mask):
        ctx.save_for_backward(q, k, v, mask)
        return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

    @staticmethod
    def backward(ctx, g):
        q, k, v, mask = ctx.saved_tensors
        prev = torch.are_deterministic_algorithms_enabled()
        prev_warn = torch.is_deterministic_algorithms_warn_only_enabled()
        torch.use_deterministic_algorithms(True)
        try:
            with torch.enable_grad():
                q_, k_, v_ = (t.detach().requires_grad_(True) for t in (q, k, v))
                out = F.scaled_dot_product_attention(q_, k_, v_, attn_mask=mask)
                gq, gk, gv = torch.autograd.grad(out, (q_, k_, v_), g)
        finally:
            torch.use_deterministic_algorithms(prev, warn_only=prev_warn)
        return gq, gk, gv, None


class PrivateDropout(nn.Module):
    """Inverted dropout whose masks come from a module-private, seeded generator."""

    def __init__(self, p: float, seed: int):
        super().__init__()
        self.p, self.seed, self._gen = float(p), int(seed), None

    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        if self._gen is None or self._gen.device != x.device:
            self._gen = torch.Generator(device=x.device)
            self._gen.manual_seed(self.seed)
        keep = torch.rand(x.shape, generator=self._gen, device=x.device, dtype=x.dtype) >= self.p
        return x * keep / (1.0 - self.p)


class MHA(nn.Module):
    """Multi-head attention with explicit W_q, W_k, W_v, W_o.

    ``impl``: 'sdpa' (fused, default of the deployed runs), 'sdpa_det' (fused forward, deterministic backward) or
    'explicit' (literal masked softmax; slow reference).
    """

    impl = "sdpa"

    def __init__(self, d: int = 64, n_heads: int = 4):
        super().__init__()
        assert d % n_heads == 0
        self.d, self.h, self.dh = d, n_heads, d // n_heads
        self.W_q = nn.Linear(d, d)
        self.W_k = nn.Linear(d, d)
        self.W_v = nn.Linear(d, d)
        self.W_o = nn.Linear(d, d)

    def forward(self, q_in, kv_in, key_valid):
        B, Lk, _ = kv_in.shape
        h, dh = self.h, self.dh
        k = self.W_k(kv_in).view(B, Lk, h, dh).transpose(1, 2)
        v = self.W_v(kv_in).view(B, Lk, h, dh).transpose(1, 2)
        q = self.W_q(q_in)
        if q.dim() == 2:
            Lq = q.shape[0]
            qh = q.view(Lq, h, dh).transpose(0, 1).unsqueeze(0).expand(B, -1, -1, -1)
        else:
            Lq = q.shape[1]
            qh = q.view(B, Lq, h, dh).transpose(1, 2)
        if self.impl in ("sdpa", "sdpa_det"):
            mask = key_valid[:, None, None, :]
            if self.impl == "sdpa_det" and torch.is_grad_enabled() and (
                    qh.requires_grad or k.requires_grad or v.requires_grad):
                out = _DetSDPA.apply(qh, k, v, mask)
            else:
                out = F.scaled_dot_product_attention(qh, k, v, attn_mask=mask)
        else:
            scores = (qh @ k.transpose(-1, -2)) / math.sqrt(dh)
            scores = scores.masked_fill(~key_valid[:, None, None, :], float("-inf"))
            out = torch.softmax(scores, dim=-1) @ v
        return self.W_o(out.transpose(1, 2).reshape(B, Lq, self.d))


class EncoderLayer(nn.Module):
    """Pre-LN self-attention layer; padding slots stay exactly 0."""

    def __init__(self, d, n_heads, ffn, dropout, seed=0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = MHA(d, n_heads)
        self.ln2 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn), nn.GELU(), nn.Linear(ffn, d))
        self.drop = PrivateDropout(dropout, seed)

    def forward(self, h, valid):
        a = self.ln1(h)
        h = h + self.drop(self.attn(a, a, valid))
        h = h + self.drop(self.ffn(self.ln2(h)))
        return h * valid.unsqueeze(-1).to(h.dtype)


class AttentionTerm(nn.Module):
    """A(c, g). phi reads [own value; ``n_nb`` neighbour values; ``n_nb`` neighbour type distances] (1 + 2 n_nb -> 32
    -> 64); the 2 n_nb neighbour columns of its first layer start at zero, so A starts as a function of the cell's own
    values only. Spatial context enters the model only through this term."""

    def __init__(self, n_genes, fm_dim=512, n_nb=15, dropout=0.1, dropout_seed=0, d=64, n_heads=4, ffn=128,
                 n_self_layers=2, z_clip=10.0):
        super().__init__()
        self.n_genes, self.d, self.z_clip, self.n_nb = n_genes, d, float(z_clip), int(n_nb)
        self.P = nn.Linear(fm_dim, d)                                    # shared gene projection (queries and tokens)
        # The deployed model first builds a Linear(1,16)-GELU-Linear(16,64) value embedding and then replaces it. The
        # throw-away module is kept so that, for a given seed, every later layer receives the same initial weights.
        _burn = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, d))
        del _burn
        self.layers = nn.ModuleList([EncoderLayer(d, n_heads, ffn, dropout, seed=1000 * dropout_seed + 101 + i)
                                     for i in range(n_self_layers)])
        self.ln_mem = nn.LayerNorm(d)
        self.ln_q = nn.LayerNorm(d)
        self.xattn = MHA(d, n_heads)
        self.ln_out = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.drop = PrivateDropout(dropout, 1000 * dropout_seed + 100)
        self.phi = nn.Sequential(nn.Linear(1 + 2 * self.n_nb, 32), nn.GELU(), nn.Linear(32, d))
        with torch.no_grad():
            self.phi[0].weight[:, 1:].zero_()
        # per-dataset value statistics: non-trainable and never carried over in a checkpoint
        self.register_buffer("z_mean", torch.zeros(n_genes), persistent=False)
        self.register_buffer("z_sd", torch.ones(n_genes), persistent=False)
        self._stats_set = False

    def set_impl(self, impl: str):
        for m in self.modules():
            if isinstance(m, MHA):
                m.impl = impl
        return self

    @torch.no_grad()
    def set_value_stats(self, mean, sd, eps: float = 1e-6):
        mean = torch.as_tensor(mean, dtype=torch.float32, device=self.z_mean.device)
        sd = torch.as_tensor(sd, dtype=torch.float32, device=self.z_sd.device)
        assert mean.shape == (self.n_genes,) and sd.shape == (self.n_genes,)
        assert torch.isfinite(mean).all() and torch.isfinite(sd).all()
        self.z_mean.copy_(mean)
        self.z_sd.copy_(sd + eps)
        self._stats_set = True

    def encode(self, V_tok, x_raw, input_mask, nb_raw=None, nb_dist=None):
        """Encode the measured, unmasked genes of each cell. ``nb_raw``: None (reference stage) or a function
        order (B, T) -> (B, n_nb, T) of unsmoothed neighbour values; ``nb_dist`` (B, n_nb) type distances."""
        assert self._stats_set, "set_value_stats() must be called before the module is used"
        B, G = input_mask.shape
        n_tok = input_mask.sum(1)
        assert bool((n_tok > 0).all()), "a cell has no tokens"
        T = int(n_tok.max())
        order = torch.argsort(input_mask.to(torch.uint8), dim=1, descending=True, stable=True)[:, :T]
        valid = torch.gather(input_mask, 1, order)
        z = ((x_raw - self.z_mean) / self.z_sd).clamp(-self.z_clip, self.z_clip)
        z = torch.where(input_mask, z, torch.zeros_like(z))
        feats = [torch.gather(z, 1, order).unsqueeze(-1)]
        if nb_raw is not None:
            xn = nb_raw(order)
            zn = ((xn - self.z_mean[order].unsqueeze(1)) / self.z_sd[order].unsqueeze(1)).clamp(-self.z_clip, self.z_clip)
            zn = torch.where(valid.unsqueeze(1), zn, torch.zeros_like(zn))
            feats.append(zn.transpose(1, 2))
            feats.append(nb_dist.to(z.dtype).unsqueeze(1).expand(B, T, self.n_nb))
        else:
            feats.append(z.new_zeros(B, T, 2 * self.n_nb))
        f = torch.cat(feats, -1)
        PV = self.P(V_tok)
        h = (PV[order] + self.phi(f)) * valid.unsqueeze(-1).to(PV.dtype)
        for layer in self.layers:
            h = layer(h, valid)
        return PV, self.ln_mem(h), valid

    def decode(self, q, mem, valid):
        """q (Q, d) projected target embeddings -> A (B, Q)."""
        y = q.unsqueeze(0) + self.drop(self.xattn(self.ln_q(q), mem, valid))
        return self.out(self.ln_out(y)).squeeze(-1)

    def forward(self, V_tok, x_raw, input_mask, nb_raw=None, nb_dist=None, V_extra=None):
        PV, mem, valid = self.encode(V_tok, x_raw, input_mask, nb_raw, nb_dist)
        A = self.decode(PV, mem, valid)
        if V_extra is None:
            return A, None
        return A, self.decode(self.P(V_extra), mem, valid)
