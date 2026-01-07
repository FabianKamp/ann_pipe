"""
ResNeXt-101-WSL model wrapper with activation extraction.
Wrapper around Facebook's WSL ResNeXt-101 implementation.
"""

import torch
from typing import Dict, List


class ResNeXtWSLWrapper(torch.nn.Module):
    """ResNeXt-101-WSL wrapper exposing intermediate activations."""

    # Layer names for activation extraction - following ResNet structure
    LAYER_NAMES: List[str] = [
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

    def __init__(self, device: torch.device | None = None):
        super().__init__()

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        # Load ResNeXt-101-WSL model
        self.model = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
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
                print(f"Registered hook for {name}")
            except (AttributeError, IndexError) as e:
                print(f"Warning: Could not find layer {name} in ResNeXt-101-WSL model: {e}")

    def _make_hook(self, identifier: str):
        def hook_fn(_module, _inputs, output):
            self.activation[identifier] = output
        return hook_fn

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """Return logits and collected activations."""
        # Clear previous activations
        self.activation.clear()
        
        # Forward pass
        logits = self.model(x)
        return logits, self.activation

    def __del__(self):
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass


def load_resnext_wsl_model(device: torch.device | None = None):
    """Factory for ResNeXt-101-WSL wrapper."""
    print("Loading ResNeXt-101-WSL model with activation extraction")
    return ResNeXtWSLWrapper(device=device)


# Layer configuration for import by setup.py
RESNEXT_WSL_LAYERS = [
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