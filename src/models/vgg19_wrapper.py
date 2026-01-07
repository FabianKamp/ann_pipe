"""
VGG19 model wrapper with activation extraction.
Wrapper around torchvision's VGG19 implementation.
"""

import torch
from typing import Dict, List
import torchvision.models as models
from torchvision.models import VGG19_Weights


class VGG19Wrapper(torch.nn.Module):
    """VGG19 wrapper exposing intermediate activations."""

    # Layer names for activation extraction - maxpool layers
    LAYER_NAMES: List[str] = ["maxpool1", "maxpool2", "maxpool3", "maxpool4", "maxpool5"]

    def __init__(self, device: torch.device | None = None):
        super().__init__()

        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        # Load VGG19 model
        self.model = models.vgg19(weights=VGG19_Weights.DEFAULT)
        self.model.to(device)
        self.model.eval()

        # Activation extraction
        self.activation: Dict[str, torch.Tensor] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self):
        """Attach forward hooks to maxpool layers."""
        # VGG19 features structure: conv layers + maxpool layers
        # We need to find the maxpool layers in the features module
        
        maxpool_count = 0
        for i, layer in enumerate(self.model.features):
            if isinstance(layer, torch.nn.MaxPool2d):
                maxpool_count += 1
                layer_name = f"maxpool{maxpool_count}"
                
                if layer_name in self.LAYER_NAMES:
                    try:
                        handle = layer.register_forward_hook(self._make_hook(layer_name))
                        self._hook_handles.append(handle)
                        print(f"Registered hook for {layer_name} at features[{i}]")
                    except Exception as e:
                        print(f"Warning: Could not register hook for {layer_name}: {e}")

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


def load_vgg19_model(device: torch.device | None = None):
    """Factory for VGG19 wrapper."""
    print("Loading VGG19 model with activation extraction")
    return VGG19Wrapper(device=device)


# Layer configuration for import by setup.py
VGG19_LAYERS = ["maxpool1", "maxpool2", "maxpool3", "maxpool4", "maxpool5"] 