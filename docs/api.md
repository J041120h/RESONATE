# API reference

## Workflow

::: resonate.prepare

::: resonate.fit

::: resonate.Resonate
    options:
      members: [predict, save, load, target_genes]

::: resonate.predict

## Data objects

::: resonate.Prepared
    options:
      members: [save, load]

::: resonate.Config

## Gene embeddings

::: resonate.GeneEmbeddings

::: resonate.load_gene_embeddings

::: resonate.extract_scgpt_spatial

::: resonate.download_embeddings

## Scoring

::: resonate.score_heldout

::: resonate.metrics.score

## Model internals

::: resonate.model.model.ResonateNet
    options:
      members: [set_data, forward, forward_extra, freeze_mu]

::: resonate.model.attention.AttentionTerm
