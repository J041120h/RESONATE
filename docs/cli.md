# Command line

```bash
resonate prepare --spatial sp.h5ad --reference ref.h5ad --embeddings emb.npz --targets genes.txt --out prep/
resonate fit     --prepared prep/ --out model/
resonate predict --prepared prep/ --model model/ --out pred.h5ad            # the fit targets
resonate predict --prepared prep/ --model model/ --genes more.txt \
                 --reference ref.h5ad --embeddings emb.npz --out pred2.h5ad   # genes outside the fit targets
resonate run     --spatial sp.h5ad --reference ref.h5ad --embeddings emb.npz --targets genes.txt --out pred.h5ad
resonate download-embeddings --name scgpt_spatial_human
```

`--config cfg.json` overrides any field of `resonate.Config` (for example `{"device": "cpu", "max_epochs": 20}`);
`--heldout-fraction 0.2` replaces `--targets` for a benchmark run. Gene files hold one symbol per line.
