"""Run the real-data Phase 8 fraud-detection experiment matrix."""

import argparse
import json
import sys

from evaluation.phase8_experiments import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_CLIENTS,
    DEFAULT_NONIID_ALPHA,
    DEFAULT_ROUNDS,
    DEFAULT_SEED,
    run_phase8,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", help="Path to the Kaggle creditcard.csv")
    parser.add_argument("--output-dir", help="Output directory (default: results/phase8)")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--clients", type=int, default=DEFAULT_CLIENTS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--noniid-alpha", type=float, default=DEFAULT_NONIID_ALPHA)
    parser.add_argument("--skip-tests", action="store_true", help="Skip the full unittest suite")
    args = parser.parse_args()
    if args.rounds < 1 or args.batch_size < 1 or args.clients < 2:
        parser.error("rounds and batch-size must be positive; clients must be at least two.")
    if args.noniid_alpha <= 0:
        parser.error("noniid-alpha must be positive.")

    try:
        result = run_phase8(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            rounds=args.rounds,
            batch_size=args.batch_size,
            seed=args.seed,
            num_clients=args.clients,
            noniid_alpha=args.noniid_alpha,
            run_tests=not args.skip_tests,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "output_dir": result["output_dir"],
        "experiments": [
            {
                "experiment_id": item["experiment_id"],
                "status": item["status"],
                "error": item.get("error"),
            }
            for item in result["experiments"]
        ],
        "graphs": result["graphs"],
        "test_summary": result["test_summary"],
        "load_error": result["load_error"],
    }, indent=2))
    return 1 if any(item["status"] != "success" for item in result["experiments"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
