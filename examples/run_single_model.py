#!/usr/bin/env python3
"""Minimal activation extraction example for a single model."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from runners.run_extract_activations import main as extract_main  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-model extraction example")
    parser.add_argument("--stimuli_root", required=True, help="Directory containing example stimuli (PNG files)")
    parser.add_argument("--folder", default=".", help="Subfolder relative to stimuli_root (default: root itself)")
    parser.add_argument("--output_dir", default="./example_output", help="Destination directory for activations")
    parser.add_argument("--model", default="CORNet-S", help="Model name to extract (default: CORNet-S)")
    parser.add_argument("--replicates", type=int, default=2, help="Replicates to run per image (default: 2)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    extract_main(
        stimuli_root=args.stimuli_root,
        output_dir=output_dir,
        folders=[args.folder],
        selected_models=[args.model],
        n_replicates=args.replicates,
        output_prefix="example_activations",
        auto_confirm=True,
    )

if __name__ == "__main__":
    main()
