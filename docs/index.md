# RESONATE

RESONATE predicts the expression of genes that were **never measured** in a spatial transcriptomics sample. It needs

* a spatial dataset (counts of the measured gene panel, cell coordinates),
* a **matched single-cell reference** that measures the genes you want to predict,
* a fixed **embedding vector for each gene** (scGPT-spatial token embeddings by default).

For a spatial cell *c* and an unmeasured gene *g* the prediction is

$$\hat x_{cg} = M_{cg} + R_{cg} + \lambda\,A_{cg}.$$

| term | what it uses |
|---|---|
| **M**, mean | the gene's overall level (an MLP of its embedding) and its mean among the 100 most similar reference cells |
| **R**, correlation | measured genes that correlate with *g* in the reference (up to 128 positive and 8 negative partners per cell type), weighted by a learned pair score and mixed over reference cell types |
| **A**, attention | a transformer over all measured genes of the cell and of its 15 spatial neighbours; $\lambda\in\{0,\tfrac14,\tfrac12,\tfrac34,1\}$ is chosen on measured genes that were set aside |

Training has two stages: pre-training on the reference, then fine-tuning on the spatial cells.

```python
import resonate
prep  = resonate.prepare(spatial, reference, embeddings, target_genes=["NPPA", "HCN1"])
model = resonate.fit(prep)
pred  = model.predict()          # AnnData, cells x target genes
```

Start with the [installation](installation.md) and the [quickstart tutorial](tutorial.ipynb), which runs on a small
MERFISH fetal-heart sample.

!!! note "Status"
    Pre-release. The package reproduces the model of the paper (see [equivalence](equivalence.md)); a cross-omics
    (RNA to protein) mode is not included yet.
