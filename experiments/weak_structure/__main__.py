import argparse
import os

# Keep each scoring process single-threaded; inference has its own fixed setting.
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

from .pipeline import DEFAULT_CONFIG, load_config


def main():
    parser = argparse.ArgumentParser(description="Independent weak-structure detectability experiment")
    parser.add_argument("command", choices=["prepare", "tune", "evaluate", "report"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", choices=["development", "validation", "test"])
    args = parser.parse_args()
    config = load_config(args.config)
    if args.command == "prepare":
        from .pipeline import prepare
        if not args.split:
            parser.error("prepare requires --split")
        prepare(config, args.split)
    elif args.command == "tune":
        from .tuning import tune
        if args.split and args.split != "validation":
            parser.error("tune only accepts validation")
        tune(config)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        evaluate(config, args.split or "test")
    else:
        from .reporting import report
        report(config, args.split or "test")


if __name__ == "__main__":
    main()
