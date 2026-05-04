#!/usr/bin/env python3
"""
Create a POD-reduced dataset from an existing ctf4science dataset.

Usage:
    python notebooks/create_pod.py <dataset> <n_modes> [--overwrite]

Example:
    python notebooks/create_pod.py msfr 12

Computes a global POD basis from all unique training matrices, projects every
train/init/test file onto the top n_modes right singular vectors, and writes
the result to data/{dataset}_pod/ with a matching YAML config.
"""

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import yaml

from ctf4science.data_module import get_config


def _load_array(path: Path) -> np.ndarray:
    """Load a single array from .npz, .mat, or .npy, returning it as float64."""
    suffix = path.suffix.lower()
    if suffix == ".npz":
        npz = np.load(path)
        key = next(iter(npz.keys()))
        return npz[key].astype(np.float64)
    if suffix == ".mat":
        import scipy.io
        mat = scipy.io.loadmat(str(path))
        keys = [k for k in mat if not k.startswith("__")]
        return mat[keys[0]].astype(np.float64)
    if suffix == ".npy":
        return np.load(path).astype(np.float64)
    raise ValueError(f"Unsupported file format: {suffix}")


def _npz_output_name(fname: str) -> str:
    """Return the output filename as .npz (changes .mat/.npy to .npz)."""
    return Path(fname).stem + ".npz"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a POD-reduced dataset.")
    parser.add_argument("dataset", help="Dataset name (e.g. msfr)")
    parser.add_argument("n_modes", type=int, help="Number of POD modes to keep")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite output directory if it already exists")
    args = parser.parse_args()

    data_root = Path(__file__).parent.parent / "data"
    src_dir = data_root / args.dataset
    dst_name = f"{args.dataset}_pod"
    dst_dir = data_root / dst_name

    if dst_dir.exists() and not args.overwrite:
        print(f"Error: {dst_dir} already exists. Use --overwrite to replace it.")
        sys.exit(1)

    config = get_config(args.dataset)
    pairs = config["pairs"]

    # ── Collect unique file sets ─────────────────────────────────────────────
    train_files: set[str] = set()
    init_files: set[str] = set()
    test_files: set[str] = set()

    for pair in pairs:
        for f in pair["train"]:
            train_files.add(f)
        if "initialization" in pair:
            init_files.add(pair["initialization"])
        test_files.add(pair["test"])

    print(f"Dataset : {args.dataset}")
    print(f"Modes   : {args.n_modes}")
    print(f"Training files : {sorted(train_files)}")
    print(f"Init files     : {sorted(init_files)}")
    print(f"Test files     : {sorted(test_files)}")

    # ── Build global POD basis from training files ───────────────────────────
    print("\nLoading training matrices for POD basis...")
    train_arrays: dict[str, np.ndarray] = {}
    all_train_chunks: list[np.ndarray] = []

    for fname in sorted(train_files):
        X = _load_array(src_dir / "train" / fname)
        train_arrays[fname] = X
        all_train_chunks.append(X)
        print(f"  Loaded {fname}: {X.shape}")

    X_all = np.concatenate(all_train_chunks, axis=0)
    mu = X_all.mean(axis=0)
    X_all_c = X_all - mu
    del X_all  # free memory before SVD

    print(f"\nComputing economy SVD on {X_all_c.shape} matrix...")
    _, S, Vt = np.linalg.svd(X_all_c, full_matrices=False)
    del X_all_c

    n_available = Vt.shape[0]
    if args.n_modes > n_available:
        print(f"Warning: requested {args.n_modes} modes but only {n_available} available. "
              f"Using {n_available}.")
        args.n_modes = n_available

    Vt_pod = Vt[: args.n_modes]
    print(f"POD basis: {Vt_pod.shape}  (kept top {args.n_modes} of {n_available} modes)")

    # ── Create output directories ────────────────────────────────────────────
    (dst_dir / "train").mkdir(parents=True, exist_ok=True)
    (dst_dir / "test").mkdir(parents=True, exist_ok=True)

    # ── Save POD modes ───────────────────────────────────────────────────────
    pod_modes_path = dst_dir / "pod_modes.npz"
    np.savez(pod_modes_path, Vt=Vt_pod.astype(np.float32),
             mu=mu.astype(np.float32), singular_values=S.astype(np.float32))
    print(f"\nSaved POD modes → {pod_modes_path}")

    # ── Project training files ───────────────────────────────────────────────
    print("\nProjecting training files...")
    for fname, X in train_arrays.items():
        coeffs = ((X - mu) @ Vt_pod.T).astype(np.float32)
        out_name = _npz_output_name(fname)
        np.savez(dst_dir / "train" / out_name, X=coeffs)
        print(f"  {fname}: {X.shape} → {coeffs.shape}")

    # ── Project initialization files ─────────────────────────────────────────
    if init_files:
        print("\nProjecting initialization files...")
    for fname in sorted(init_files):
        X = _load_array(src_dir / "train" / fname)
        coeffs = ((X - mu) @ Vt_pod.T).astype(np.float32)
        out_name = _npz_output_name(fname)
        np.savez(dst_dir / "train" / out_name, X=coeffs)
        print(f"  {fname}: {X.shape} → {coeffs.shape}")

    # ── Project test files ───────────────────────────────────────────────────
    print("\nProjecting test files...")
    for fname in sorted(test_files):
        X = _load_array(src_dir / "test" / fname)
        coeffs = ((X - mu) @ Vt_pod.T).astype(np.float32)
        out_name = _npz_output_name(fname)
        np.savez(dst_dir / "test" / out_name, X=coeffs)
        print(f"  {fname}: {X.shape} → {coeffs.shape}")

    # ── Write new YAML ───────────────────────────────────────────────────────
    new_config = copy.deepcopy(config)

    meta = new_config["metadata"]
    orig_dim = meta.get("spatial_dimension", None)
    meta["spatial_dimension"] = args.n_modes
    meta["n_pod_modes"] = args.n_modes
    meta["original_dataset"] = args.dataset
    meta["original_spatial_dimension"] = orig_dim
    meta["pod_modes_file"] = "pod_modes.npz"

    if "matrix_shapes" in meta:
        for key in list(meta["matrix_shapes"].keys()):
            shape = meta["matrix_shapes"][key]
            new_key = _npz_output_name(key)
            meta["matrix_shapes"][new_key] = [shape[0], args.n_modes]
            if new_key != key:
                del meta["matrix_shapes"][key]

    if "matrix_start_index" in meta:
        for key in list(meta["matrix_start_index"].keys()):
            new_key = _npz_output_name(key)
            if new_key != key:
                meta["matrix_start_index"][new_key] = meta["matrix_start_index"].pop(key)

    # Update pair file references to .npz if source used a different format
    for pair in new_config["pairs"]:
        pair["train"] = [_npz_output_name(f) for f in pair["train"]]
        pair["test"] = _npz_output_name(pair["test"])
        if "initialization" in pair:
            pair["initialization"] = _npz_output_name(pair["initialization"])

    yaml_path = dst_dir / f"{dst_name}.yaml"
    with open(yaml_path, "w") as fh:
        yaml.dump(new_config, fh, default_flow_style=False, sort_keys=False)

    print(f"\nWrote config → {yaml_path}")

    if "num_fields" in new_config.get("evaluation_params", {}):
        print("Note: evaluation_params.num_fields and modes were copied from the original.")
        print("      Review them in the new YAML — they may not apply in POD coefficient space.")

    print(f"\nDone. New dataset '{dst_name}' at {dst_dir}")
    print(f"  Dimensions: {orig_dim} → {args.n_modes}")


if __name__ == "__main__":
    main()
