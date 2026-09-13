"""Entry point for the TC-GT scripts.

    python main.py --mode train --csv data/slides.csv --graph_dir /path --cohort lung --fold 0
    python main.py --mode ph --csv data/slides.csv --graph_dir /path --cohort lung
    python main.py --mode local_topo --csv data/slides.csv --graph_dir /path --cohort lung
"""
import argparse
import sys


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description="TC-GT")
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "ph", "local_topo"],
        help="train | precompute graph-level PH | precompute node-level topology",
    )
    args, rest = parser.parse_known_args(argv)

    if args.mode == "train":
        from train import main as run
    elif args.mode == "ph":
        from compute_ph import main as run
    else:
        from compute_local_topo import main as run
    return run(rest)


if __name__ == "__main__":
    main()
