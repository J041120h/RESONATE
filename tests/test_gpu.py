import numpy as np
import pytest
import torch

import resonate
from resonate import Config

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]


def test_gpu_default_numerics_run_and_match_cpu_closely(toy, toy_prep, toy_cfg):
    """Paper numerics (TF32, bf16 attention, fused kernel) on the GPU give predictions close to fp32 CPU for the same weights."""
    cfg_gpu = Config(**{**toy_cfg.to_dict(), "device": "cuda", "tf32": True, "attn_bf16": True})
    m = resonate.fit(toy_prep, cfg_gpu, verbose=False)
    p_gpu = m.predict(as_anndata=False)[0]
    assert np.isfinite(p_gpu).all()
    m.net.attn_bf16 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    p32 = m.predict(as_anndata=False)[0]
    assert np.corrcoef(p_gpu.ravel(), p32.ravel())[0, 1] > 0.99
