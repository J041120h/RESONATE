import numpy as np
import pandas as pd
import pytest

from resonate import GeneEmbeddings, load_gene_embeddings
from resonate.embeddings import cosine_matrix


def test_roundtrip_npz(tmp_path):
    e = GeneEmbeddings(["A", "B", "C"], np.eye(3, 4), "t")
    e.save(tmp_path / "e.npz")
    f = load_gene_embeddings(tmp_path / "e.npz")
    assert f.genes == ["A", "B", "C"] and np.allclose(f.matrix, e.matrix) and f.dim == 4


def test_csv_and_dataframe(tmp_path):
    df = pd.DataFrame(np.arange(6, dtype=float).reshape(3, 2), index=["A", "B", "C"])
    df.to_csv(tmp_path / "e.csv")
    assert load_gene_embeddings(tmp_path / "e.csv").genes == ["A", "B", "C"]
    assert load_gene_embeddings(df).dim == 2


def test_panel_oov_gets_mean_and_zero_cosine():
    e = GeneEmbeddings(["A", "B"], np.array([[1.0, 0.0], [0.0, 1.0]]))
    V, oov = e.panel(["A", "X", "B"])
    assert oov.tolist() == [False, True, False]
    assert np.allclose(V[1], [0.5, 0.5])
    S = cosine_matrix(V, oov)
    assert S[1].sum() == 0 and S[:, 1].sum() == 0 and S[0, 0] == pytest.approx(1.0)


def test_bad_inputs(tmp_path):
    with pytest.raises(ValueError):
        GeneEmbeddings(["A", "A"], np.ones((2, 2)))
    with pytest.raises(ValueError):
        GeneEmbeddings(["A"], np.array([[np.nan]]))
    with pytest.raises(FileNotFoundError):
        load_gene_embeddings(tmp_path / "missing.npz")
    with pytest.raises(ValueError):
        GeneEmbeddings(["A"], np.ones((1, 2))).panel(["X", "Y"])
