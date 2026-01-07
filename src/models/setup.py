"""
Functions for initializing and configuring neural network models.
Includes model configurations and layer mappings.
"""
import torch
import torchvision.models as models
from torchvision.models import VGG19_Weights, ResNet50_Weights, ResNeXt101_32X8D_Weights
from cornet import cornet_s

from .vit_clip import load_vit_clip_model, VIT_CLIP_LAYERS
from .vit_16 import load_vit_16_model, VIT_16_LAYERS
from .convnext_clip import load_convnext_clip_model, CONVNEXT_CLIP_LAYERS
from .resnet_clip import load_resnet_clip_model, RESNET_CLIP_LAYERS
from .vit_b32_clip import load_vit_b32_clip_model, VIT_B32_CLIP_LAYERS
from .vit_b16_laion_clip import load_vit_b16_laion_clip_model, VIT_B16_LAION_CLIP_LAYERS
from .vgg19_wrapper import load_vgg19_model, VGG19_LAYERS
from .resnext_wsl_wrapper import load_resnext_wsl_model, RESNEXT_WSL_LAYERS
from .timm_models import (
    load_convnext_large_model, load_deit_base_model,
    load_vit_b32_timm_model, CONVNEXT_LARGE_LAYERS, DEIT_BASE_LAYERS,
    VIT_B32_TIMM_LAYERS
)

# Layer configurations for models without custom wrapper scripts
# (Models with custom wrappers import their layer configs from respective modules)

CORNET_S_LAYERS = ["V1", "V2", "V4", "IT"]
CORNET_RT_LAYERS = ["V1", "V2", "V4", "IT"]
CORNET_Z_LAYERS = ["V1", "V2", "V4", "IT"]

# For models that need all layers extracted (not just the 5 standard ones)
ALEXNET_LAYERS = [
    "features.0",   # conv1
    "features.3",   # conv2
    "features.5",   # conv3
    "features.8",   # conv4
    "features.10"   # conv5
]

VGG19_LAYERS = [
    "maxpool1",     # First maxpool (after conv1_2)
    "maxpool2",     # Second maxpool (after conv2_2)
    "maxpool3",     # Third maxpool (after conv3_4)
    "maxpool4",     # Fourth maxpool (after conv4_4)
    "maxpool5"      # Fifth maxpool (after conv5_4)
]

RESNET50_LAYERS = [
    "conv1",         # initial convolutional layer
    "layer1.0.conv1",  # first conv in first block of layer1
    "layer1.0.conv2",  # second conv in first block of layer1
    "layer1.1.conv1",  # first conv in second block of layer1
    "layer1.1.conv2",  # second conv in second block of layer1
    "layer1.2.conv1",  # first conv in third block of layer1
    "layer1.2.conv2",  # second conv in third block of layer1
    "layer2.0.conv1",  # first conv in first block of layer2
    "layer2.0.conv2",  # second conv in first block of layer2
    "layer2.1.conv1",  # first conv in second block of layer2
    "layer2.1.conv2",  # second conv in second block of layer2
    "layer2.2.conv1",  # first conv in third block of layer2
    "layer2.2.conv2",  # second conv in third block of layer2
    "layer2.3.conv1",  # first conv in fourth block of layer2
    "layer2.3.conv2",  # second conv in fourth block of layer2
    "layer3.0.conv1",  # first conv in first block of layer3
    "layer3.0.conv2",  # second conv in first block of layer3
    "layer3.1.conv1",  # first conv in second block of layer3
    "layer3.1.conv2",  # second conv in second block of layer3
    "layer3.2.conv1",  # first conv in third block of layer3
    "layer3.2.conv2",  # second conv in third block of layer3
    "layer3.3.conv1",  # first conv in fourth block of layer3
    "layer3.3.conv2",  # second conv in fourth block of layer3
    "layer3.4.conv1",  # first conv in fifth block of layer3
    "layer3.4.conv2",  # second conv in fifth block of layer3
    "layer3.5.conv1",  # first conv in sixth block of layer3
    "layer3.5.conv2",  # second conv in sixth block of layer3
    "layer4.0.conv1",  # first conv in first block of layer4
    "layer4.0.conv2",  # second conv in first block of layer4
    "layer4.1.conv1",  # first conv in second block of layer4
    "layer4.1.conv2",  # second conv in second block of layer4
    "layer4.2.conv1",  # first conv in third block of layer4
    "layer4.2.conv2"   # second conv in third block of layer4
]

RESNEXT_LAYERS = [
    "conv1",           # stem conv output
    "layer1.0.conv1",  # first conv in first block of layer1
    "layer1.0.conv2",  # second conv in first block of layer1
    "layer1.1.conv1",  # first conv in second block of layer1
    "layer1.1.conv2",  # second conv in second block of layer1
    "layer1.2.conv1",  # first conv in third block of layer1
    "layer1.2.conv2",  # second conv in third block of layer1
    "layer2.0.conv1",  # first conv in first block of layer2
    "layer2.0.conv2",  # second conv in first block of layer2
    "layer2.1.conv1",  # first conv in second block of layer2
    "layer2.1.conv2",  # second conv in second block of layer2
    "layer2.2.conv1",  # first conv in third block of layer2
    "layer2.2.conv2",  # second conv in third block of layer2
    "layer2.3.conv1",  # first conv in fourth block of layer2
    "layer2.3.conv2",  # second conv in fourth block of layer2
    "layer3.0.conv1",  # first conv in first block of layer3
    "layer3.0.conv2",  # second conv in first block of layer3
    "layer3.1.conv1",  # first conv in second block of layer3
    "layer3.1.conv2",  # second conv in second block of layer3
    "layer3.2.conv1",  # first conv in third block of layer3
    "layer3.2.conv2",  # second conv in third block of layer3
    "layer3.3.conv1",  # first conv in fourth block of layer3
    "layer3.3.conv2",  # second conv in fourth block of layer3
    "layer3.4.conv1",  # first conv in fifth block of layer3
    "layer3.4.conv2",  # second conv in fifth block of layer3
    "layer3.5.conv1",  # first conv in sixth block of layer3
    "layer3.5.conv2",  # second conv in sixth block of layer3
    "layer4.0.conv1",  # first conv in first block of layer4
    "layer4.0.conv2",  # second conv in first block of layer4
    "layer4.1.conv1",  # first conv in second block of layer4
    "layer4.1.conv2",  # second conv in second block of layer4
    "layer4.2.conv1",  # first conv in third block of layer4
    "layer4.2.conv2"   # second conv in third block of layer4
]

EFFICIENTNET_B7_LAYERS = [
    "features.0",       # stem conv (Conv2dNormActivation)
    "features.1.0", "features.1.1", "features.1.2",  # Stage 1: 3 MBConv blocks
    "features.2.0", "features.2.1", "features.2.2", "features.2.3", "features.2.4", "features.2.5",  # Stage 2: 6 MBConv blocks
    "features.3.0", "features.3.1", "features.3.2", "features.3.3", "features.3.4", "features.3.5",  # Stage 3: 6 MBConv blocks
    "features.4.0", "features.4.1", "features.4.2", "features.4.3", "features.4.4", "features.4.5", "features.4.6", "features.4.7", "features.4.8",  # Stage 4: 9 MBConv blocks
    "features.5.0", "features.5.1", "features.5.2", "features.5.3", "features.5.4", "features.5.5", "features.5.6", "features.5.7", "features.5.8",  # Stage 5: 9 MBConv blocks
    "features.6.0", "features.6.1", "features.6.2", "features.6.3", "features.6.4", "features.6.5", "features.6.6", "features.6.7", "features.6.8", "features.6.9", "features.6.10", "features.6.11",  # Stage 6: 12 MBConv blocks
    "features.7.0", "features.7.1", "features.7.2",  # Stage 7: 3 MBConv blocks
]

VIT_L_16_LAYERS = [
    "transformer.blocks.0.mlp.fc2",
    "transformer.blocks.1.mlp.fc2",
    "transformer.blocks.2.mlp.fc2",
    "transformer.blocks.3.mlp.fc2",
    "transformer.blocks.4.mlp.fc2",
    "transformer.blocks.5.mlp.fc2",
    "transformer.blocks.6.mlp.fc2",
    "transformer.blocks.7.mlp.fc2",
    "transformer.blocks.8.mlp.fc2",
    "transformer.blocks.9.mlp.fc2",
    "transformer.blocks.10.mlp.fc2",
    "transformer.blocks.11.mlp.fc2",
    "transformer.blocks.12.mlp.fc2",
    "transformer.blocks.13.mlp.fc2",
    "transformer.blocks.14.mlp.fc2",
    "transformer.blocks.15.mlp.fc2",
    "transformer.blocks.16.mlp.fc2",
    "transformer.blocks.17.mlp.fc2",
    "transformer.blocks.18.mlp.fc2",
    "transformer.blocks.19.mlp.fc2",
    "transformer.blocks.20.mlp.fc2",
    "transformer.blocks.21.mlp.fc2",
    "transformer.blocks.22.mlp.fc2",
    "transformer.blocks.23.mlp.fc2"
]

 
# Layer mapping for standardization - maps layer names to standard depth indices (1-5)
# Only includes layers that are actually extracted by models
LAYER_MAPPING = {
    # Layer 1 (V1 equivalent) - Early processing
    "V1": 1,  # CORNet family
    "blocks.4.attn.qkv": 1,  # ViT-CLIP V1
    "transformer.blocks.0.pwff.fc2": 1,  # ViT-16 V1
    "stages.0.blocks.2.conv_dw": 1,  # ConvNeXt-Large V1
    "blocks.2.mlp.fc2": 1,  # DeiT V1
    "blocks.1.mlp.fc2": 1,  # ViT-B-32 timm V1
    "layer1.2.conv3": 1,  # ResNet CLIP V1
    "trunk.stages.0.blocks.2.conv_dw": 1,  # ConvNeXt CLIP V1
    "transformer.resblocks.2.mlp.c_fc": 1,  # ViT-B32 CLIP V1
    "transformer.resblocks.1.mlp.c_fc": 1,  # ViT-B16 LAION CLIP V1
    
    # Layer 2 (V2 equivalent) - Early-mid processing
    "V2": 2,  # CORNet family
    "blocks.8.attn": 2,  # ViT-CLIP V2
    "transformer.blocks.1.pwff.fc2": 2,  # ViT-16 V2
    "stages.1.blocks.2.conv_dw": 2,  # ConvNeXt-Large V2
    "blocks.5.mlp.fc2": 2,  # DeiT V2
    "blocks.3.mlp.fc2": 2,  # ViT-B-32 timm V2
    "layer2.3.conv3": 2,  # ResNet CLIP V2
    "trunk.stages.1.blocks.2.conv_dw": 2,  # ConvNeXt CLIP V2
    "transformer.resblocks.5.mlp.c_fc": 2,  # ViT-B32 CLIP V2
    "transformer.resblocks.2.mlp.c_fc": 2,  # ViT-B16 LAION CLIP V2
    
    # Layer 3 (V4 equivalent) - Mid processing
    "V4": 3,  # CORNet family
    "blocks.5.attn": 3,  # ViT-CLIP V4
    "transformer.blocks.2.pwff.fc2": 3,  # ViT-16 V4
    "stages.2.blocks.8.conv_dw": 3,  # ConvNeXt-Large V4
    "blocks.8.mlp.fc2": 3,  # DeiT V4
    "blocks.6.mlp.fc2": 3,  # ViT-B-32 timm V4
    "layer3.5.conv3": 3,  # ResNet CLIP V4
    "trunk.stages.2.blocks.8.conv_dw": 3,  # ConvNeXt CLIP V4
    "transformer.resblocks.8.mlp.c_fc": 3,  # ViT-B32 CLIP V4
    "transformer.resblocks.5.mlp.c_fc": 3,  # ViT-B16 LAION CLIP V4
    
    # Layer 4 (IT equivalent) - Mid-late processing
    "IT": 4,  # CORNet family
    "blocks.9.attn": 4,  # ViT-CLIP IT
    "transformer.blocks.3.pwff.fc2": 4,  # ViT-16 IT
    "stages.2.blocks.17.conv_dw": 4,  # ConvNeXt-Large IT
    "blocks.10.mlp.fc2": 4,  # DeiT IT
    "blocks.9.mlp.fc2": 4,  # ViT-B-32 timm IT
    "layer4.1.conv3": 4,  # ResNet CLIP IT
    "trunk.stages.2.blocks.17.conv_dw": 4,  # ConvNeXt CLIP IT
    "transformer.resblocks.10.mlp.c_fc": 4,  # ViT-B32 CLIP IT
    "transformer.resblocks.8.mlp.c_fc": 4,  # ViT-B16 LAION CLIP IT
    
    # Layer 5 - Late processing
    "blocks.11.attn": 5,  # ViT-CLIP late
    "transformer.blocks.6.pwff.fc2": 5,  # ViT-16 late
    "stages.3.blocks.2.conv_dw": 5,  # ConvNeXt-Large late
    "blocks.11.mlp.fc2": 5,  # DeiT late
    "blocks.7.mlp.fc2": 5,  # ViT-B-32 timm late
    "layer4.2.conv3": 5,  # ResNet CLIP late
    "trunk.stages.3.blocks.2.conv_dw": 5,  # ConvNeXt CLIP late
    "transformer.resblocks.11.mlp.c_fc": 5,  # ViT-B32 CLIP late
    "transformer.resblocks.10.mlp.c_fc": 5,  # ViT-B16 LAION CLIP late
}

# Model configurations - uses imported layer configurations from model scripts
MODEL_CONFIGS = {
    # Category 1: Vision CNNs (ImageNet Training)
    "ResNet50": {
        "layers": RESNET50_LAYERS,
        "model_type": "resnet50"
    },
    "ConvNeXt-Large": {
        "layers": CONVNEXT_LARGE_LAYERS,  # Imported from timm_models
        "model_type": "convnext_large"
    },
    "CORNet-S": {
        "layers": CORNET_S_LAYERS,
        "model_type": "cornet"
    },
    "VGG19": {
        "layers": VGG19_LAYERS,  # Imported from vgg19_wrapper
        "model_type": "vgg19_wrapper"
    },
    
    # Category 2: Vision Transformers (ImageNet Training)
    "ViT-B-16": {
        "layers": VIT_16_LAYERS,  # Imported from vit_16
        "model_type": "vit16"
    },
    "DeiT-Base": {
        "layers": DEIT_BASE_LAYERS,  # Imported from timm_models
        "model_type": "deit"
    },
    "ViT-B-32-timm": {
        "layers": VIT_B32_TIMM_LAYERS,  # Imported from timm_models
        "model_type": "vit_b32_timm"
    },
    
    # Category 3: Vision-Language CNNs (CLIP/WSL Training)
    "ResNeXt-101-WSL": {
        "layers": RESNEXT_WSL_LAYERS,  # Imported from resnext_wsl_wrapper
        "model_type": "resnext_wsl_wrapper"
    },
    "ResNet50-CLIP": {
        "layers": RESNET_CLIP_LAYERS,  # Imported from resnet_clip
        "model_type": "resnet_clip"
    },
    "ConvNeXt-Large-CLIP": {
        "layers": CONVNEXT_CLIP_LAYERS,  # Imported from convnext_clip
        "model_type": "convnext_clip"
    },
    
    # Category 4: Vision-Language Transformers (CLIP Training)
    "ViT-B-16-CLIP": {
        "layers": VIT_CLIP_LAYERS,  # Imported from vit_clip
        "model_type": "vitclip"
    },
    "ViT-B-32-CLIP": {
        "layers": VIT_B32_CLIP_LAYERS,  # Imported from vit_b32_clip
        "model_type": "vit_b32_clip"
    },
    "ViT-B-16-LAION-CLIP": {
        "layers": VIT_B16_LAION_CLIP_LAYERS,  # Imported from vit_b16_laion_clip
        "model_type": "vit_b16_laion_clip"
    }
}

def get_models_config():
    """
    Get model configurations without initializing the models.
    
    Returns:
        dict: Dictionary containing model configurations with layer information
    """
    return MODEL_CONFIGS

def initialize_models(device=None, selected_models=None):
    """
    Initialize models and move them to the specified device.
    
    Args:
        device (torch.device, optional): Device to move models to. If None, uses CUDA if available.
        selected_models (list, optional): List of model names to initialize. If None, initializes all models.
        
    Returns:
        dict: Dictionary of initialized models and their configurations
    """
    # Enhanced device selection: prefer Apple Silicon GPU (mps) over CPU
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get base configuration
    models_config = get_models_config()
    
    # Filter models if selected_models is provided
    if selected_models is not None:
        print(f"Initializing selected models: {', '.join(selected_models)}")
        models_config = {k: v for k, v in models_config.items() if k in selected_models}
    
    # Define model loader functions to avoid loading unused models
    model_loaders = {
        # Category 1: Vision CNNs (ImageNet Training)
        "ResNet50": lambda: models.resnet50(weights=ResNet50_Weights.DEFAULT),
        "ConvNeXt-Large": lambda: load_convnext_large_model(device=device),
        "CORNet-S": lambda: cornet_s(pretrained=True, map_location=device).to(device),
        "VGG19": lambda: load_vgg19_model(device=device),
        
        # Category 2: Vision Transformers (ImageNet Training)
        "ViT-B-16": lambda: load_vit_16_model(device=device),
        "DeiT-Base": lambda: load_deit_base_model(device=device),
        "ViT-B-32-timm": lambda: load_vit_b32_timm_model(device=device),
        
        # Category 3: Vision-Language CNNs (CLIP/WSL Training)
        "ResNeXt-101-WSL": lambda: load_resnext_wsl_model(device=device),
        "ResNet50-CLIP": lambda: load_resnet_clip_model(device=device),
        "ConvNeXt-Large-CLIP": lambda: load_convnext_clip_model(device=device),
        
        # Category 4: Vision-Language Transformers (CLIP Training)
        "ViT-B-16-CLIP": lambda: load_vit_clip_model(device=device),
        "ViT-B-32-CLIP": lambda: load_vit_b32_clip_model(device=device),
        "ViT-B-16-LAION-CLIP": lambda: load_vit_b16_laion_clip_model(device=device)
    }
    
    # Initialize only the selected models
    for model_name in models_config:
        if model_name in model_loaders:
            print(f"Loading {model_name}...")
            models_config[model_name]["model"] = model_loaders[model_name]()
    
    # Set all models to evaluation mode and move to device
    for model_config in models_config.values():
        if "model" in model_config:
            model_config["model"].to(device)
            model_config["model"].eval()
    
    return models_config

# Categories for DNN alignment curves
CATEGORIES = {
    "Vision CNNs": ["ResNet50", "ConvNeXt-Large", "CORNet-S", "VGG19"],
    "Vision ViTs": ["ViT-B-16", "DeiT-Base", "ViT-B-32-timm"],
    "Vision-Language CNNs": ["ResNeXt-101-WSL", "ResNet50-CLIP", "ConvNeXt-Large-CLIP"],
    "Vision-Language ViTs": ["ViT-B-16-CLIP", "ViT-B-32-CLIP", "ViT-B-16-LAION-CLIP"]
}