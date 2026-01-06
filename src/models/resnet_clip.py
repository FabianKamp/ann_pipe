"""
ResNet50 CLIP model loading and configuration.
Wrapper around open_clip's ResNet50 implementation.
"""

import torch
from typing import Dict, List

try:
    import open_clip
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "`open_clip` library is required to load ResNet CLIP. Install via\n"
        "  pip install open-clip-torch\n"
    ) from err


class ResNetCLIPWrapper(torch.nn.Module):
    """ResNet50 CLIP wrapper exposing intermediate activations."""

    DEFAULT_MODEL_NAME = "RN50"
    DEFAULT_PRETRAINED = "openai"

    # Layer names for activation extraction - all conv layers from all residual blocks
    LAYER_NAMES: List[str] = []
    
    # ResNet50 architecture: layer1(3 blocks), layer2(4 blocks), layer3(6 blocks), layer4(3 blocks)
    @classmethod
    def _generate_layer_names(cls):
        layer_names = []
        layer_blocks = {"layer1": 3, "layer2": 4, "layer3": 6, "layer4": 3}
        for layer_name, num_blocks in layer_blocks.items():
            for block_idx in range(num_blocks):
                for conv_idx in [1, 2, 3]:  # conv1, conv2, conv3 in each residual block
                    layer_names.append(f"{layer_name}.{block_idx}.conv{conv_idx}")
        return layer_names
    
    def __init__(self, model_name: str | None = None, pretrained: str | None = None, device: torch.device | None = None):
        # Generate comprehensive layer names
        self.__class__.LAYER_NAMES = self._generate_layer_names()
        super().__init__()

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        if model_name is None:
            model_name = self.DEFAULT_MODEL_NAME
        if pretrained is None:
            pretrained = self.DEFAULT_PRETRAINED

        # Load model
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.visual  # Extract visual encoder
        self.model.to(device)
        self.model.eval()

        # Activation extraction
        self.activation: Dict[str, torch.Tensor] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
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
        for name in self.LAYER_NAMES:
            path_parts = name.split(".")
            try:
                layer = self._resolve_submodule(self.model, path_parts)
                handle = layer.register_forward_hook(self._make_hook(name))
                self._hook_handles.append(handle)
            except (AttributeError, IndexError):
                print(f"Warning: Could not find layer {name} in ResNet CLIP model")

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


def load_resnet_clip_model(model_name: str | None = None, pretrained: str | None = None, device: torch.device | None = None):
    """Factory for ResNet CLIP wrapper."""
    if model_name is None:
        model_name = ResNetCLIPWrapper.DEFAULT_MODEL_NAME
    if pretrained is None:
        pretrained = ResNetCLIPWrapper.DEFAULT_PRETRAINED

    print(f"Loading ResNet CLIP model: {model_name} with {pretrained}")
    return ResNetCLIPWrapper(model_name=model_name, pretrained=pretrained, device=device)


# Layer configuration for import by setup.py - all conv layers from all residual blocks
RESNET_CLIP_LAYERS = []
_layer_blocks = {"layer1": 3, "layer2": 4, "layer3": 6, "layer4": 3}
for _layer_name, _num_blocks in _layer_blocks.items():
    for _block_idx in range(_num_blocks):
        for _conv_idx in [1, 2, 3]:  # conv1, conv2, conv3 in each residual block
            RESNET_CLIP_LAYERS.append(f"{_layer_name}.{_block_idx}.conv{_conv_idx}") 