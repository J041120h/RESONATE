"""All tunable settings, with the defaults of the deployed model (paper Methods and Supplementary Methods)."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Config:
    # ---- preparation (20 gene folds and K = 100 reference neighbours are fixed by the model) ----
    smooth_weight: float = 0.3          # x~ = 0.3 x + 0.7 mean of 15 neighbours
    k_smooth: int = 15                  # neighbours used for smoothing
    n_spatial_nb: int = 15              # spatial neighbours in the attention features
    n_pos_partners: int = 128
    n_neg_partners: int = 8
    n_types_range: tuple = (13, 18)     # Leiden resolution is bisected to land in this range
    n_types_target: int = 15
    min_type_cells: int = 50            # smaller clusters are pooled into one type
    shrink_n0: int = 200                # type correlation shrinkage toward the pooled within-type correlation
    n_aux_genes: int = 2000
    aux_per_batch: int = 256
    split_fractions: tuple = (0.8, 0.1, 0.1)   # used only when the spatial data carry no split column
    # ---- model -------------------------------------------------------------------------------------------------
    dropout: float = 0.1
    mu_hidden: tuple = (96, 96, 96)
    corr_hidden: int = 32
    # ---- training ----------------------------------------------------------------------------------------------
    seed: int = 0
    max_epochs: int = 500
    patience: int = 15
    stage1_batch_size: int = 4096
    stage1_lr: float = 4e-3
    stage2_batch_size: int = 1024
    stage2_lr: float = 2e-3
    mask_frac_cell: float = 0.20        # genes hidden independently in each cell (loss weight 0.6)
    mask_frac_batch: float = 0.05       # genes hidden in every cell of a batch (loss weight 0.4)
    pseudo_held_fraction: float = 0.2   # measured genes set aside to choose the checkpoint and lambda
    lambda_grid: tuple = (0.0, 0.25, 0.5, 0.75, 1.0)
    mu_warmup_steps: int = 200
    mu_warmup_lr: float = 1e-3
    # ---- numerics / hardware -----------------------------------------------------------------------------------
    device: str | None = None           # None = cuda if available
    tf32: bool = True
    attn_impl: str = "sdpa"             # 'sdpa' | 'sdpa_det' | 'explicit'
    attn_bf16: bool = True              # bf16 autocast inside the attention term (CUDA only)
    data_on_host: bool = False          # keep the (cells x genes) matrices in host RAM (very large datasets)

    def to_dict(self):
        return asdict(self)
