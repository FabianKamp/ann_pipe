#!/usr/bin/env python3
"""Example script demonstrating how to read extracted activations."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data.lazy_activations import (
    load_lazy_activations,
    get_available_models,
    get_available_images,
    get_available_layers,
    get_activation_shape,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example: reading extracted activations")
    parser.add_argument("h5_file", help="Path to HDF5 activation file")
    parser.add_argument("--model", default=None, help="Model to inspect (default: first available)")
    parser.add_argument("--show_layers", action="store_true", help="Display all layer names")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    print(f"Inspecting: {args.h5_file}\n")
    
    # Get available models
    models = get_available_models(args.h5_file)
    print(f"Available models ({len(models)}):")
    for model in models:
        print(f"  - {model}")
    
    # Select model to inspect
    model_name = args.model if args.model else models[0]
    print(f"\nInspecting model: {model_name}")
    
    # Get images for this model
    images = get_available_images(args.h5_file, model_name)
    print(f"Number of images: {len(images)}")
    if images:
        print(f"Sample images: {', '.join(images[:5])}")
        if len(images) > 5:
            print(f"  ... and {len(images) - 5} more")
    
    # Get layers
    layers = get_available_layers(args.h5_file, model_name)
    print(f"\nNumber of layers: {len(layers)}")
    
    if args.show_layers:
        print("All layers:")
        for layer in layers:
            print(f"  - {layer}")
    else:
        print(f"Sample layers: {', '.join(layers[:3])}")
        if len(layers) > 3:
            print(f"  ... and {len(layers) - 3} more (use --show_layers to see all)")
    
    # Get activation shape
    if images and layers:
        shape = get_activation_shape(args.h5_file, model_name)
        print(f"\nActivation shape: {shape}")
        print(f"  - n_replicates: {shape[0]}")
        print(f"  - feature_dim: {shape[1]}")
    
    # Demonstrate lazy loading
    print("\n" + "="*60)
    print("Lazy loading example:")
    print("="*60)
    
    acts = load_lazy_activations(args.h5_file)
    
    if images and layers:
        sample_img = images[0]
        sample_layer = layers[0]
        
        print(f"Loading: {model_name} / {sample_img} / {sample_layer}")
        activation = acts[model_name][sample_img][sample_layer]
        print(f"Loaded activation shape: {activation.shape}")
        print(f"Data type: {activation.dtype}")
        print(f"Memory: {activation.nbytes / 1024:.2f} KB")
        
        # Show statistics
        print(f"\nActivation statistics:")
        print(f"  Mean: {activation.mean():.4f}")
        print(f"  Std:  {activation.std():.4f}")
        print(f"  Min:  {activation.min():.4f}")
        print(f"  Max:  {activation.max():.4f}")
    
    acts.close()
    print("\nDone!")


if __name__ == "__main__":
    main()

