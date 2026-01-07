"""
Timm model wrappers for activation extraction.
Handles ConvNeXt, DeiT, Swin, and LeViT models.
"""

import torch
from typing import Dict, List

try:
    import timm
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "`timm` library is required to load these models. Install via\n"
        "  pip install timm\n"
    ) from err


class TimmModelWrapper(torch.nn.Module):
    """Generic timm model wrapper exposing intermediate activations."""

    def __init__(self, model_name: str, layer_names: List[str], device: torch.device | None = None):
        super().__init__()

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        # Load model
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.to(device)
        self.model.eval()

        # Activation extraction
        self.activation: Dict[str, torch.Tensor] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self.layer_names = layer_names
        self._register_hooks()

    def _resolve_submodule(self, root: torch.nn.Module, path: List[str]):
        """Resolve dotted path into a real sub-Module."""
        module: torch.nn.Module = root
        for part in path:
            if part.isdigit():
                module = module[int(part)]
            else:
                module = getattr(module, part)
        return module

    def _register_hooks(self):
        """Attach forward hooks to pre-defined layers."""
        for name in self.layer_names:
            path_parts = name.split(".")
            try:
                layer = self._resolve_submodule(self.model, path_parts)
                handle = layer.register_forward_hook(self._make_hook(name))
                self._hook_handles.append(handle)
            except (AttributeError, IndexError):
                print(f"Warning: Could not find layer {name} in model")

    def _make_hook(self, identifier: str):
        def hook_fn(_module, _inputs, output):
            self.activation[identifier] = output
        return hook_fn

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """Return logits and collected activations."""
        logits = self.model(x)
        return logits, self.activation

    def __del__(self):
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass


# Model-specific factory functions

def load_convnext_large_model(device: torch.device | None = None):
    """Factory for ConvNeXt-Large model."""
    print("Loading ConvNeXt-Large model from timm")
    
    # Create the model first to inspect its actual layer structure
    temp_model = timm.create_model("convnext_large", pretrained=False)
    
    # Extract layer names from the actual model structure
    layer_names = []
    stage_blocks = [3, 3, 27, 3]  # blocks per stage for ConvNeXt-Large
    
    # Look for actual layer names in the model structure
    for stage_idx, num_blocks in enumerate(stage_blocks):
        for block_idx in range(num_blocks):
            # Try different possible naming conventions for ConvNeXt layers
            possible_names = [
                f"stages.{stage_idx}.blocks.{block_idx}.conv_dw",
                f"stages.{stage_idx}.blocks.{block_idx}.dwconv",
                f"stages.{stage_idx}.blocks.{block_idx}.depthwise_conv",
                f"stages.{stage_idx}.blocks.{block_idx}.gamma",
                f"stages.{stage_idx}.blocks.{block_idx}.norm",
                f"stages.{stage_idx}.blocks.{block_idx}.pwconv1",
                f"stages.{stage_idx}.blocks.{block_idx}.pwconv2"
            ]
            
            # Find which layer name actually exists in the model
            for name in possible_names:
                if hasattr_nested(temp_model, name):
                    layer_names.append(name)
                    break
            else:
                # If none of the expected names work, add a fallback
                print(f"Warning: Could not find expected layer at stage {stage_idx}, block {block_idx}")
                # Add the first name as fallback - the wrapper will handle the error gracefully
                layer_names.append(f"stages.{stage_idx}.blocks.{block_idx}.pwconv2")
    
    # Clean up temporary model
    del temp_model
    
    return TimmModelWrapper("convnext_large", layer_names, device)


def hasattr_nested(obj, attr_path):
    """Check if nested attribute exists in object (e.g., 'stages.0.blocks.0.conv_dw')."""
    attrs = attr_path.split('.')
    current_obj = obj
    try:
        for attr in attrs:
            if attr.isdigit():
                current_obj = current_obj[int(attr)]
            else:
                current_obj = getattr(current_obj, attr)
        return True
    except (AttributeError, IndexError, KeyError):
        return False


def load_deit_base_model(device: torch.device | None = None):
    """Factory for DeiT-Base model."""
    layer_names = [f"blocks.{i}.mlp.fc2" for i in range(12)]  # DeiT-Base: 12 transformer blocks
    print("Loading DeiT-Base model from timm")
    return TimmModelWrapper("deit_base_patch16_224", layer_names, device)


def load_swin_base_model(device: torch.device | None = None):
    """Factory for Swin-Base model."""
    # Swin-Base: Layer 0(2 blocks), Layer 1(2 blocks), Layer 2(18 blocks), Layer 3(2 blocks)
    layer_names = []
    layer_blocks = [2, 2, 18, 2]  # blocks per layer
    for layer_idx, num_blocks in enumerate(layer_blocks):
        for block_idx in range(num_blocks):
            layer_names.append(f"layers.{layer_idx}.blocks.{block_idx}.mlp.fc2")
    
    print("Loading Swin-Base model from timm")
    return TimmModelWrapper("swin_base_patch4_window7_224", layer_names, device)


def load_levit_128_model(device: torch.device | None = None):
    """Factory for LeViT-128s model."""
    # LeViT-128s: Stage 0 (2 blocks), Stage 1 (3 blocks), Stage 2 (4 blocks) = 9 total blocks
    layer_names = []
    blocks_per_stage = [2, 3, 4]  # Actual architecture of LeViT-128s
    for stage_idx, num_blocks in enumerate(blocks_per_stage):
        for block_idx in range(num_blocks):
            layer_names.append(f"stages.{stage_idx}.blocks.{block_idx}")
    
    print("Loading LeViT-128s model from timm")
    return TimmModelWrapper("levit_128s", layer_names, device)


def load_vit_b32_timm_model(device: torch.device | None = None):
    """Factory for ViT-B-32 model from timm."""
    layer_names = [f"blocks.{i}.mlp.fc2" for i in range(12)]  # ViT-B-32: 12 transformer blocks
    print("Loading ViT-B-32 model from timm")
    return TimmModelWrapper("vit_base_patch32_224", layer_names, device)


# Layer configurations for import by setup.py
# ConvNeXt-Large: comprehensive extraction from all stages and blocks  
CONVNEXT_LARGE_LAYERS = []
_stage_blocks = [3, 3, 27, 3]  # blocks per stage
for _stage_idx, _num_blocks in enumerate(_stage_blocks):
    for _block_idx in range(_num_blocks):
        # Use conv_dw which exists (verified in validation)
        CONVNEXT_LARGE_LAYERS.append(f"stages.{_stage_idx}.blocks.{_block_idx}.conv_dw")

DEIT_BASE_LAYERS = [f"blocks.{i}.mlp.fc2" for i in range(12)]  # DeiT-Base: 12 transformer blocks

# Swin-Base: comprehensive extraction from all layers and blocks
SWIN_BASE_LAYERS = []
_layer_blocks = [2, 2, 18, 2]  # blocks per layer
for _layer_idx, _num_blocks in enumerate(_layer_blocks):
    for _block_idx in range(_num_blocks):
        SWIN_BASE_LAYERS.append(f"layers.{_layer_idx}.blocks.{_block_idx}.mlp.fc2")

# LeViT-128s: comprehensive extraction from all stages and blocks
LEVIT_128_LAYERS = []
_blocks_per_stage = [2, 3, 4]  # Actual architecture of LeViT-128s
for _stage_idx, _num_blocks in enumerate(_blocks_per_stage):
    for _block_idx in range(_num_blocks):
        LEVIT_128_LAYERS.append(f"stages.{_stage_idx}.blocks.{_block_idx}")

VIT_B32_TIMM_LAYERS = [f"blocks.{i}.mlp.fc2" for i in range(12)]  # ViT-B-32: 12 transformer blocks 