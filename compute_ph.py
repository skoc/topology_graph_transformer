"""Precompute graph-level persistence images.

    python compute_ph.py --csv data/slides.csv --graph_dir /path/to/bins --out data/ph

Writes <cohort>_ph.npz, one (F, ph_dim) array per slide keyed by slide id.
"""
import argparse
import os

import numpy as np
from dgl.data.utils import load_graphs

from dataset import load_slides, resolve_graph
from ph import graph_ph_features
from topology import FILTRATIONS


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Precompute graph-level PH features")
    p.add_argument("--csv", default="data/slides.csv")
    p.add_argument("--graph_dir", required=True)
    p.add_argument("--out", default="data/ph")
    p.add_argument("--cohort", default="lung", choices=["lung", "kidney"])
    p.add_argument("--filtrations", nargs="+", default=list(FILTRATIONS))
    p.add_argument("--resolution", type=int, default=20)
    p.add_argument("--bandwidth", type=float, default=1.0)
    p.add_argument("--tau", type=float, default=1.0, help="HKS diffusion time")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    df = load_slides(args.csv, cohort=args.cohort, task="typing")
    resolution = (args.resolution, args.resolution)

    features, missing, failed = {}, 0, 0
    for i, row in enumerate(df.itertuples(), start=1):
        path = resolve_graph(args.graph_dir, row.slide_id)
        if path is None:
            missing += 1
            continue
        try:
            graph = load_graphs(path)[0][0]
            features[row.slide_id] = np.stack([
                graph_ph_features(graph, f, args.tau, resolution, args.bandwidth)
                for f in args.filtrations
            ])
        except Exception as exc:
            failed += 1
            print(f"failed {row.slide_id}: {exc}")
        if i % 50 == 0:
            print(f"{i}/{len(df)} slides")

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"{args.cohort}_ph.npz")
    np.savez_compressed(out_path, **features)
    print(f"wrote {len(features)} slides to {out_path} (missing {missing}, failed {failed})")


if __name__ == "__main__":
    main()
