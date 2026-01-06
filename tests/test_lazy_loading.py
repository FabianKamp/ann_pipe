#!/usr/bin/env python3
"""Unit tests for lazy activation loading utilities."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import h5py
import numpy as np
import tempfile
import os


def create_test_h5(path: str) -> None:
    """Create a minimal test HDF5 file with expected structure."""
    with h5py.File(path, "w") as f:
        # Create structure: {model}/{image}/{layer}
        for model in ["ResNet50", "CORNet-S"]:
            for img in ["object_01", "object_02"]:
                for layer in ["layer1", "layer2"]:
                    # Shape: (n_replicates, feature_dim)
                    data = np.random.randn(4, 64).astype(np.float32)
                    f.create_dataset(f"{model}/{img}/{layer}", data=data)


def test_lazy_loading() -> None:
    """Test lazy loading functionality."""
    from src.data.lazy_activations import (
        load_lazy_activations,
        get_available_models,
        get_available_images,
        get_available_layers,
        get_activation_shape,
    )
    
    # Create temporary test file
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        test_file = tmp.name
    
    try:
        create_test_h5(test_file)
        
        # Test inspection utilities
        models = get_available_models(test_file)
        assert set(models) == {"ResNet50", "CORNet-S"}, f"Expected 2 models, got {models}"
        
        images = get_available_images(test_file, "ResNet50")
        assert set(images) == {"object_01", "object_02"}, f"Expected 2 images, got {images}"
        
        layers = get_available_layers(test_file, "ResNet50")
        assert set(layers) == {"layer1", "layer2"}, f"Expected 2 layers, got {layers}"
        
        shape = get_activation_shape(test_file, "ResNet50")
        assert shape == (4, 64), f"Expected shape (4, 64), got {shape}"
        
        # Test lazy loading
        acts = load_lazy_activations(test_file)
        
        # Test dictionary-like access
        assert "ResNet50" in acts
        assert "object_01" in acts["ResNet50"]
        assert "layer1" in acts["ResNet50"]["object_01"]
        
        # Test data loading
        data = acts["ResNet50"]["object_01"]["layer1"]
        assert data.shape == (4, 64), f"Expected shape (4, 64), got {data.shape}"
        assert data.dtype == np.float32
        
        # Test keys
        assert set(acts.keys()) == {"ResNet50", "CORNet-S"}
        assert set(acts["ResNet50"].keys()) == {"object_01", "object_02"}
        assert set(acts["ResNet50"]["object_01"].keys()) == {"layer1", "layer2"}
        
        # Test cache clearing
        acts["ResNet50"]["object_01"].clear_cache()
        acts["ResNet50"].clear_cache()
        acts.close()
        
        print("All lazy loading tests passed!")
        
    finally:
        # Clean up
        if os.path.exists(test_file):
            os.unlink(test_file)


if __name__ == "__main__":
    test_lazy_loading()

