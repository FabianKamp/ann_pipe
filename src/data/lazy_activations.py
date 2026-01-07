"""
Lazy-loading activation dictionary classes for memory-efficient access to HDF5 activation data.
Provides the same API as regular nested dictionaries but only loads data when accessed.
"""
import h5py
import numpy as np
from typing import Dict, List


class LazyLayerDict:
    """Dictionary-like access to layer activations within an image, loaded on-demand."""
    
    def __init__(self, h5_path: str, model_name: str, img_name: str):
        """
        Args:
            h5_path: Path to HDF5 file
            model_name: Name of the model
            img_name: Name of the image
        """
        self.h5_path = h5_path
        self.model_name = model_name
        self.img_name = img_name
        self._cache = {}  # Optional: cache recently accessed layers
        
        # Get available layers
        with h5py.File(h5_path, "r") as f:
            self._layer_names = list(f[model_name][img_name].keys())
    
    def __getitem__(self, layer_name: str) -> np.ndarray:
        """Load and return activation data for the specified layer."""
        if layer_name in self._cache:
            return self._cache[layer_name]
            
        if layer_name not in self._layer_names:
            raise KeyError(f"Layer {layer_name} not found in image {self.img_name}")
        
        # Load data from HDF5
        with h5py.File(self.h5_path, "r") as f:
            data = f[self.model_name][self.img_name][layer_name][:]
        
        # Optional: cache small activations (< 10MB)
        if data.nbytes < 10 * 1024 * 1024:
            self._cache[layer_name] = data
            
        return data
    
    def __contains__(self, layer_name: str) -> bool:
        """Check if layer exists in this image."""
        return layer_name in self._layer_names
    
    def keys(self):
        """Return available layer names."""
        return self._layer_names
    
    def clear_cache(self):
        """Clear the layer activation cache to free memory."""
        self._cache.clear()


class LazyModelDict:
    """Dictionary-like access to image activations within a model, loaded on-demand."""
    
    def __init__(self, h5_path: str, model_name: str):
        """
        Args:
            h5_path: Path to HDF5 file
            model_name: Name of the model
        """
        self.h5_path = h5_path
        self.model_name = model_name
        self._cache = {}  # Cache LazyLayerDict instances
        
        # Get available images
        with h5py.File(h5_path, "r") as f:
            self._img_names = list(f[model_name].keys())
    
    def __getitem__(self, img_name: str) -> LazyLayerDict:
        """Return lazy layer dictionary for the specified image."""
        if img_name in self._cache:
            return self._cache[img_name]
            
        if img_name not in self._img_names:
            raise KeyError(f"Image {img_name} not found in model {self.model_name}")
        
        layer_dict = LazyLayerDict(self.h5_path, self.model_name, img_name)
        self._cache[img_name] = layer_dict
        return layer_dict
    
    def __contains__(self, img_name: str) -> bool:
        """Check if image exists in this model."""
        return img_name in self._img_names
    
    def keys(self):
        """Return available image names."""
        return self._img_names
    
    def clear_cache(self):
        """Clear the image cache to free memory."""
        for layer_dict in self._cache.values():
            layer_dict.clear_cache()
        self._cache.clear()


class LazyActivationDict:
    """Top-level lazy activation dictionary that mimics the original nested dict structure."""
    
    def __init__(self, h5_path: str):
        """
        Args:
            h5_path: Path to HDF5 file containing activations
        """
        self.h5_path = h5_path
        self._cache = {}  # Cache LazyModelDict instances
        
        # Get model names
        try:
            with h5py.File(h5_path, "r") as f:
                self._model_names = list(f.keys())
        except Exception as e:
            raise ValueError(f"Cannot open HDF5 file {h5_path}: {e}")
    
    def __getitem__(self, model_name: str) -> LazyModelDict:
        """Return lazy model dictionary for the specified model."""
        if model_name in self._cache:
            return self._cache[model_name]
            
        if model_name not in self._model_names:
            raise KeyError(f"Model {model_name} not found in activation file")
        
        model_dict = LazyModelDict(self.h5_path, model_name)
        self._cache[model_name] = model_dict
        return model_dict
    
    def __contains__(self, model_name: str) -> bool:
        """Check if model exists in activations."""
        return model_name in self._model_names
    
    def keys(self):
        """Return available model names."""
        return self._model_names
    
    def close(self):
        """Clear all caches and free memory."""
        for model_dict in self._cache.values():
            model_dict.clear_cache()
        self._cache.clear()


# Convenience functions
def load_lazy_activations(h5_path: str) -> LazyActivationDict:
    """
    Load activations using lazy loading from HDF5 file.
    
    This provides dictionary-like access to activations without loading everything into memory.
    Data is only loaded when accessed, making it suitable for large activation datasets.
    
    Args:
        h5_path: Path to HDF5 activation file
        
    Returns:
        LazyActivationDict: Dictionary-like object with lazy loading
        
    Example:
        >>> acts = load_lazy_activations("model_activations_semantic.h5")
        >>> resnet_acts = acts["ResNet50"]["object_01"]["layer1.0.conv1"]  # Only loads this layer
    """
    return LazyActivationDict(h5_path)


def load_model_activations_lazy(h5_path: str, model_name: str) -> Dict[str, LazyModelDict]:
    """
    Load activations for a specific model using lazy loading.
    
    Args:
        h5_path: Path to HDF5 activation file
        model_name: Name of model to load
        
    Returns:
        Dict with single model using lazy loading
        
    Example:
        >>> acts = load_model_activations_lazy("activations.h5", "ResNet50")
        >>> layer_acts = acts["ResNet50"]["object_01"]["layer4.2.conv3"]
    """
    lazy_all = LazyActivationDict(h5_path)
    return {model_name: lazy_all[model_name]}


# Utility functions for inspecting HDF5 structure
def get_available_models(h5_path: str) -> List[str]:
    """
    Get list of all models stored in HDF5 activation file.
    
    Args:
        h5_path: Path to HDF5 activation file
        
    Returns:
        List of model names
    """
    with h5py.File(h5_path, "r") as f:
        return list(f.keys())


def get_available_images(h5_path: str, model_name: str) -> List[str]:
    """
    Get list of all image IDs for a specific model.
    
    Args:
        h5_path: Path to HDF5 activation file
        model_name: Name of the model
        
    Returns:
        List of image IDs
    """
    with h5py.File(h5_path, "r") as f:
        if model_name not in f:
            raise KeyError(f"Model {model_name} not found in {h5_path}")
        return list(f[model_name].keys())


def get_available_layers(h5_path: str, model_name: str, sample_image: str = None) -> List[str]:
    """
    Get list of all layer names for a specific model.
    
    Args:
        h5_path: Path to HDF5 activation file
        model_name: Name of the model
        sample_image: Image ID to sample from (if None, uses first image)
        
    Returns:
        List of layer names
    """
    with h5py.File(h5_path, "r") as f:
        if model_name not in f:
            raise KeyError(f"Model {model_name} not found in {h5_path}")
        
        if sample_image is None:
            # Use first available image
            images = list(f[model_name].keys())
            if not images:
                raise ValueError(f"No images found for model {model_name}")
            sample_image = images[0]
        
        if sample_image not in f[model_name]:
            raise KeyError(f"Image {sample_image} not found in model {model_name}")
        
        return list(f[model_name][sample_image].keys())


def get_activation_shape(h5_path: str, model_name: str, sample_image: str = None, 
                         layer_name: str = None) -> tuple:
    """
    Get the shape of activation arrays for a model/layer.
    
    Args:
        h5_path: Path to HDF5 activation file
        model_name: Name of the model
        sample_image: Image ID to sample from (if None, uses first image)
        layer_name: Layer name (if None, uses first layer)
        
    Returns:
        Tuple representing activation shape (n_replicates, feature_dim)
    """
    with h5py.File(h5_path, "r") as f:
        if model_name not in f:
            raise KeyError(f"Model {model_name} not found in {h5_path}")
        
        if sample_image is None:
            images = list(f[model_name].keys())
            if not images:
                raise ValueError(f"No images found for model {model_name}")
            sample_image = images[0]
        
        if sample_image not in f[model_name]:
            raise KeyError(f"Image {sample_image} not found in model {model_name}")
        
        if layer_name is None:
            layers = list(f[model_name][sample_image].keys())
            if not layers:
                raise ValueError(f"No layers found for model {model_name}, image {sample_image}")
            layer_name = layers[0]
        
        if layer_name not in f[model_name][sample_image]:
            raise KeyError(f"Layer {layer_name} not found")
        
        return f[model_name][sample_image][layer_name].shape

