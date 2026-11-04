"""Command line: ``resonate prepare | fit | predict | run | download-embeddings | version``.

All commands read and write AnnData (``.h5ad``) and plain folders; there are no cluster-specific paths."""
from __future__ import annotations

import argparse
import json
import sys


def _config(args):
    from .config import Config
    kw = {}
    if getattr(args, "config", None):
        kw = json.load(open(args.config))
    for k in ("device", "seed", "max_epochs", "patience"):
        v = getattr(args, k, None)
        if v is not None:
            kw[k] = v
    return Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in kw.items()})


def _add_common(p):
    p.add_argument("--config", help="JSON file overriding any field of resonate.Config")
    p.add_argument("--device", help="cpu | cuda (default: cuda if available)")
    p.add_argument("--seed", type=int)


def _prepare(args):
    import anndata as ad
    from . import prepare
    cfg = _config(args)
    genes = [g.strip() for g in open(args.targets).read().split() if g.strip()] if args.targets else None
    prep = prepare(ad.read_h5ad(args.spatial), ad.read_h5ad(args.reference), args.embeddings, target_genes=genes,
                   heldout_fraction=args.heldout_fraction, heldout_seed=args.heldout_seed, section_key=args.section_key,
                   batch_key=args.batch_key, label_key=args.label_key, config=cfg)
    prep.save(args.out)
    print(prep)


def _fit(args):
    from . import Prepared, fit
    prep = Prepared.load(args.prepared)
    model = fit(prep, _config(args) if (args.config or args.device or args.seed is not None
                                       or args.max_epochs is not None or args.patience is not None) else None,
                max_train_cells=args.max_train_cells)
    model.save(args.out)
    print(json.dumps(model.summary["stage2"], indent=1, default=str))


def _predict(args):
    import anndata as ad
    from . import Prepared, Resonate
    prep = Prepared.load(args.prepared)
    model = Resonate.load(args.model, prep, device=args.device)
    genes = [g.strip() for g in open(args.genes).read().split() if g.strip()] if args.genes else None
    ref = ad.read_h5ad(args.reference) if args.reference else None
    out = model.predict(genes, reference=ref, gene_embeddings=args.embeddings)
    out.write_h5ad(args.out)
    print(f"wrote {args.out}: {out.n_obs} cells x {out.n_vars} genes")


def _run(args):
    import anndata as ad
    from . import fit, prepare
    cfg = _config(args)
    genes = [g.strip() for g in open(args.targets).read().split() if g.strip()] if args.targets else None
    prep = prepare(ad.read_h5ad(args.spatial), ad.read_h5ad(args.reference), args.embeddings, target_genes=genes,
                   heldout_fraction=args.heldout_fraction, section_key=args.section_key, batch_key=args.batch_key,
                   label_key=args.label_key, config=cfg)
    model = fit(prep, cfg)
    model.predict().write_h5ad(args.out)
    print(f"wrote {args.out}")


def _download(args):
    from .embeddings import download_embeddings
    print(download_embeddings(args.name, args.dest))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="resonate", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def data_args(p):
        p.add_argument("--spatial", required=True, help="spatial .h5ad (raw counts, obsm['spatial'])")
        p.add_argument("--reference", required=True, help="matched single-cell reference .h5ad (raw counts)")
        p.add_argument("--embeddings", required=True, help="gene-embedding table (.npz / .csv)")
        p.add_argument("--targets", help="text file, one gene per line: genes to predict")
        p.add_argument("--heldout-fraction", type=float, help="benchmark mode: hold out this fraction of measured genes")
        p.add_argument("--heldout-seed", type=int, default=0)
        p.add_argument("--section-key", default="section")
        p.add_argument("--batch-key", default="donor")
        p.add_argument("--label-key")
        _add_common(p)

    p = sub.add_parser("prepare", help="build model inputs"); data_args(p); p.add_argument("--out", required=True)
    p.set_defaults(fn=_prepare)
    p = sub.add_parser("fit", help="train on prepared inputs"); p.add_argument("--prepared", required=True)
    p.add_argument("--out", required=True); p.add_argument("--max-epochs", dest="max_epochs", type=int)
    p.add_argument("--patience", type=int); p.add_argument("--max-train-cells", type=int, default=0)
    _add_common(p); p.set_defaults(fn=_fit)
    p = sub.add_parser("predict", help="write predicted expression to .h5ad")
    p.add_argument("--prepared", required=True); p.add_argument("--model", required=True)
    p.add_argument("--out", required=True); p.add_argument("--genes", help="text file, one gene per line (default: fit targets)")
    p.add_argument("--reference"); p.add_argument("--embeddings"); p.add_argument("--device")
    p.set_defaults(fn=_predict)
    p = sub.add_parser("run", help="prepare + fit + predict in one go"); data_args(p)
    p.add_argument("--out", required=True); p.set_defaults(fn=_run)
    p = sub.add_parser("download-embeddings"); p.add_argument("--name", default="scgpt_spatial_human")
    p.add_argument("--dest"); p.set_defaults(fn=_download)
    p = sub.add_parser("version"); p.set_defaults(fn=lambda a: print(__import__("resonate").__version__))
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
