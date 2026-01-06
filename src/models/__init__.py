"""
Model imports for the abstraction_ventral_models package.

This file makes the models directory a proper Python package and provides
convenient imports for commonly used components.
"""

# Core model setup functions
from .setup import get_models_config, initialize_models

# Model loaders
from .slip import load_slip_model
from .vit_clip import load_vit_clip_model
from .vit_16 import load_vit_16_model

__all__ = [
    'get_models_config',
    'initialize_models',
    'load_vit_clip_model',
    'load_vit_16_model',
    'load_slip_model',
] 