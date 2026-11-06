"""Command-line options reach ``prepare`` (fast: ``prepare`` is replaced by a stub)."""
import anndata
import pytest

import resonate
from resonate import cli


class _Stop(Exception):
    pass


@pytest.mark.parametrize("command", ["prepare", "run"])
def test_heldout_options_reach_prepare(command, monkeypatch):
    seen = {}

    def fake_prepare(*args, **kw):
        seen.update(kw)
        raise _Stop

    monkeypatch.setattr(anndata, "read_h5ad", lambda path: None)
    monkeypatch.setattr(resonate, "prepare", fake_prepare)
    with pytest.raises(_Stop):
        cli.main([command, "--spatial", "sp.h5ad", "--reference", "ref.h5ad", "--embeddings", "emb.npz",
                  "--heldout-fraction", "0.2", "--heldout-seed", "3", "--out", "out"])
    assert seen["heldout_fraction"] == 0.2 and seen["heldout_seed"] == 3
