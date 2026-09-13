"""Precompute node-level slice sequences and the pairwise merge heights.

    python compute_local_topo.py --csv data/slides.csv --graph_dir /path/to/bins \
        --out data/local_topo

Writes <cohort>_toposcan.npz (per-node slice tokens, feeds LTA) and
<cohort>_merge.npz (c_ij, feeds the topology-aware attention bias).
"""
import argparse
import os

import numpy as np
from dgl.data.utils import load_graphs

from dataset import load_slides, resolve_graph
from local_topo import local_toposcan_sequence
from topology import filtration_values, merge_height_matrix, undirected_edges


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Precompute node-level topology caches")
    p.add_argument("--csv", default="data/slides.csv")
    p.add_argument("--graph_dir", required=True)
    p.add_argument("--out", default="data/local_topo")
    p.add_argument("--cohort", default="lung", choices=["lung", "kidney"])
    p.add_argument("--k", type=int, default=2, help="hops of the induced subgraph")
    p.add_argument("--num_slices", type=int, default=5)
    p.add_argument("--window", type=int, default=2)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--tau", type=float, default=1.0, help="HKS diffusion time")
    p.add_argument("--skip_merge", action="store_true")
    p.add_argument("--skip_toposcan", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    df = load_slides(args.csv, cohort=args.cohort, task="typing")

    toposcan, merge, missing, failed = {}, {}, 0, 0
    for i, row in enumerate(df.itertuples(), start=1):
        path = resolve_graph(args.graph_dir, row.slide_id)
        if path is None:
            missing += 1
            continue
        try:
            graph = load_graphs(path)[0][0]
            if not args.skip_toposcan:
                toposcan[row.slide_id] = local_toposcan_sequence(
                    graph,
                    k=args.k,
                    num_slices=args.num_slices,
                    window=args.window,
                    stride=args.stride,
                    tau=args.tau,
                )
            if not args.skip_merge:
                values = filtration_values(graph, "hks", tau=args.tau)
                c = merge_height_matrix(values, undirected_edges(graph))
                merge[row.slide_id] = c.astype(np.float16)  # (N, N) per slide
        except Exception as exc:
            failed += 1
            print(f"failed {row.slide_id}: {exc}")
        if i % 50 == 0:
            print(f"{i}/{len(df)} slides")

    os.makedirs(args.out, exist_ok=True)
    if toposcan:
        path = os.path.join(args.out, f"{args.cohort}_toposcan.npz")
        np.savez_compressed(path, **toposcan)
        print(f"wrote {len(toposcan)} slides to {path}")
    if merge:
        path = os.path.join(args.out, f"{args.cohort}_merge.npz")
        np.savez_compressed(path, **merge)
        print(f"wrote {len(merge)} slides to {path}")
    print(f"missing {missing}, failed {failed}")


if __name__ == "__main__":
    main()
