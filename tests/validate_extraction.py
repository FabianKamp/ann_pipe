#!/usr/bin/env python3
"""Smoke test for verifying activation extraction output structure."""
from __future__ import annotations

import argparse
import h5py
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from runners.run_extract_activations import main as extract_main  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate extraction pipeline on a small dataset")
    parser.add_argument("--stimuli_root", required=True, help="Directory with validation stimuli")
    parser.add_argument("--folder", default=".", help="Subfolder relative to stimuli_root to process")
    parser.add_argument("--output_dir", default="./validation_output", help="Where to store validation HDF5")
    parser.add_argument("--models", nargs="+", default=["CORNet-S"], help="Models to run for validation")
    parser.add_argument("--replicates", type=int, default=2, help="Replicates per image for validation")
    return parser.parse_args()


def run_validation(args: argparse.Namespace) -> Path:
    os.makedirs(args.output_dir, exist_ok=True)
    extract_main(
        stimuli_root=args.stimuli_root,
        output_dir=args.output_dir,
        folders=[args.folder],
        selected_models=args.models,
        n_replicates=args.replicates,
        output_prefix="validation",
        auto_confirm=True,
    )
    sanitized = args.folder.strip("./").replace(os.sep, "_") or "root"
    return Path(args.output_dir) / f"validation_{sanitized}.h5"


def inspect_file(h5_path: Path, expected_models: list[str], expected_layers: dict[str, list[str]] | None = None) -> None:
    if not h5_path.exists():
        raise FileNotFoundError(f"Expected output file missing: {h5_path}")

    with h5py.File(h5_path, "r") as handle:
        for model in expected_models:
            if model not in handle:
                raise KeyError(f"Model group {model} missing in {h5_path}")
            model_group = handle[model]
            sample_images = list(model_group.keys())
            if not sample_images:
                raise ValueError(f"No images stored for model {model}")
            first_image = sample_images[0]
            layers = list(model_group[first_image].keys())
            if not layers:
                raise ValueError(f"No layers stored for model {model} image {first_image}")
            if expected_layers and model in expected_layers:
                missing = set(expected_layers[model]) - set(layers)
                if missing:
                    raise ValueError(f"Missing layers for {model}: {sorted(missing)}")
            dataset = model_group[first_image][layers[0]][:]
            if dataset.ndim != 2:
                raise ValueError(f"Dataset has unexpected shape {dataset.shape} for {model}:{layers[0]}")


def main() -> None:
    args = parse_args()
    h5_path = run_validation(args)
    inspect_file(h5_path, expected_models=args.models)
    print(f"Validation succeeded. Output saved to {h5_path}")


if __name__ == "__main__":
    main()
