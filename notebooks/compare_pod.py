#!/usr/bin/env python3
"""
Evaluate POD reconstruction quality against the original test data.

For each test pair, loads the POD coefficients from {dataset}_pod, reconstructs
to full space using the saved POD modes, and runs the standard ctf4science metrics
against the original test data.

Usage:
    python notebooks/compare_pod.py [--dataset msfr] [--pairs 1 2 3 ...]

Example:
    python notebooks/compare_pod.py
    python notebooks/compare_pod.py --pairs 1 2 8
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ctf4science.data_module import get_config
from ctf4science.eval_module import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate POD reconstruction quality.")
    parser.add_argument("--dataset", default="msfr", help="Source dataset name (default: msfr)")
    parser.add_argument("--pairs", type=int, nargs="+", help="Pair IDs to evaluate (default: all)")
    args = parser.parse_args()

    pod_dataset = f"{args.dataset}_pod"
    data_root = Path(__file__).parent.parent / "data"
    pod_dir = data_root / pod_dataset

    modes_path = pod_dir / "pod_modes.npz"
    if not modes_path.exists():
        print(f"Error: {modes_path} not found. Run create_pod.py first.")
        sys.exit(1)

    modes = np.load(modes_path)
    Vt = modes["Vt"].astype(np.float64)   # [r, D]
    mu = modes["mu"].astype(np.float64)   # [D]
    print(f"Loaded POD modes: Vt {Vt.shape}, mu {mu.shape}")

    pod_config = get_config(pod_dataset)
    pairs = pod_config["pairs"]

    if args.pairs:
        pairs = [p for p in pairs if p["id"] in args.pairs]
        if not pairs:
            print(f"Error: no pairs matched {args.pairs}")
            sys.exit(1)

    all_results: dict[int, dict[str, float]] = {}

    for pair in sorted(pairs, key=lambda p: p["id"]):
        pair_id = pair["id"]
        test_fname = pair["test"]

        coeffs_path = pod_dir / "test" / test_fname
        coeffs = np.load(coeffs_path)["X"].astype(np.float64)   # [N, r]
        X_rec = (coeffs @ Vt + mu).astype(np.float32)           # [N, D]

        results = evaluate(args.dataset, pair_id, X_rec)
        all_results[pair_id] = results
        print(f"  Pair {pair_id}: {results}")

    print(f"\n{'Pair':<6}{'Metric':<18}{'Score':>8}")
    print(f"{'-'*6}{'-'*18}{'-'*8}")
    for pair_id, metrics in sorted(all_results.items()):
        for metric, score in metrics.items():
            print(f"{pair_id:<6}{metric:<18}{score:>8.2f}")


if __name__ == "__main__":
    main()
