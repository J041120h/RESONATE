# How to choose a reference and embeddings

## The reference

The reference is where RESONATE learns what a gene looks like: its level, its co-expression partners and how it varies
between cell types. It must

1. **measure the genes you want to predict**, with raw counts and the same gene-naming scheme (symbols) as the spatial panel;
2. **come from the same tissue**, ideally the same donors or disease state. The paper's six benchmark datasets use
   tissue-matched references;
3. contain enough cells: a few thousand is workable, tens of thousands is better. Cell types smaller than 50 reference
   training cells fall back to the pooled correlation structure and are pooled into one type if their Leiden cluster has
   fewer than 50 cells;
4. have a batch label if it pools donors or samples (`obs["donor"]`, argument `batch_key`). The reference cell types and the
   cell-type mixtures are Harmony-corrected by it; without it, cell types are computed without correction.

You do not need cell-type labels: reference types are Leiden clusters of the reference (resolution bisected to give 13 to 18
types). Author labels can be used for coloring and for stratifying the split (`label_key`).

Single-nucleus references work (the paper uses them for adult heart and cortex); check that the target genes have
reasonable detection (at least 1 % of cells) because the correlation structure of rarely detected genes is noisy.

## The embeddings

Each gene needs one fixed vector. The default, and what the paper uses, is the **scGPT-spatial** token-embedding table
(512 dimensions, human gene symbols).

| option | how | notes |
|---|---|---|
| scGPT-spatial (default) | `resonate.extract_scgpt_spatial("scGPT_spatial_v1/")` or the downloadable table | trained on spatial data; check overlap of its pretraining sources with your dataset (a caveat for every evaluation) |
| your own table | `resonate.load_gene_embeddings(path_or_dataframe)` | rows = gene symbols, any dimension; for example Geneformer token embeddings or a reference-derived PCA of gene profiles |
| none for some genes | automatic | panel genes without a vector share one learnable vector; their cosine to other genes is 0; **target genes without an embedding cannot be predicted** at inference |

The paper's embedding ablation (Geneformer 10M/104M/316M, norm-matched random, reference PCA, crossed with reference size)
is reported in the manuscript's supplement; read it before substituting another table. Mouse data needs a mouse-symbol table (the
default human table will cover only genes with identical symbols).

Gene symbols must match exactly (case included) between the spatial data, the reference and the embedding table. Convert
Ensembl IDs to symbols first.
