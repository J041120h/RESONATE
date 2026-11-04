import pytest

from resonate import Config
from resonate.toy import make_toy_data


@pytest.fixture(scope="session")
def toy():
    return make_toy_data()


@pytest.fixture(scope="session")
def toy_cfg():
    """Small, CPU-friendly settings; the model and the training rules are unchanged."""
    return Config(device="cpu", n_types_range=(3, 6), n_types_target=4, min_type_cells=20, shrink_n0=50,
                  n_aux_genes=100, aux_per_batch=32, max_epochs=3, patience=3, stage1_batch_size=256,
                  stage2_batch_size=256, tf32=False, attn_bf16=False)


@pytest.fixture(scope="session")
def toy_prep(toy, toy_cfg):
    import resonate
    sp, ref, emb = toy
    return resonate.prepare(sp, ref, emb, heldout_fraction=0.2, config=toy_cfg, verbose=False)


@pytest.fixture(scope="session")
def toy_net(toy_prep, toy_cfg):
    """An untrained model on the toy inputs, set up like fine-tuning (stage 2) but with lookups = raw."""
    import numpy as np
    from resonate.train import build_net, partner_sets
    p = toy_prep
    net = build_net(p, toy_cfg, partner_sets(p, toy_cfg), "cpu", 0)
    L = np.nan_to_num(p.r_ref_cv)
    net.set_data(stage=2, X=p.X, X_in=p.X_input, votes=p.votes_sp, pi=p.pi, L_train=L, L_test=p.r_ref, NB=p.NB,
                 labels=p.labels_sp, device="cpu")
    from resonate.train import set_value_stats
    set_value_stats(net, p.X, p.tr, p.vis)
    net.freeze_mu()
    return net
