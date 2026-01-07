"""
ViT-B-16 LAION CLIP model loading and configuration.
Wrapper around open_clip's ViT-B-16 implementation with LAION-2B pretrained weights.
"""

import torch
from typing import Dict, List

try:
    import open_clip
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "`open_clip` library is required to load ViT-B-16 LAION CLIP. Install via\n"
        "  pip install open-clip-torch\n"
    ) from err


class ViTB16LAIONCLIPWrapper(torch.nn.Module):
    """ViT-B-16 LAION CLIP wrapper exposing intermediate activations."""

    DEFAULT_MODEL_NAME = "ViT-B-16"
    DEFAULT_PRETRAINED = "laion2b_s34b_b88k"

    # Layer names for activation extraction - all transformer blocks
    LAYER_NAMES: List[str] = [f"transformer.resblocks.{i}.mlp.c_fc" for i in range(12)]  # ViT-B-16: 12 blocks

    def __init__(self, model_name: str | None = None, pretrained: str | None = None, device: torch.device | None = None):
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
                print(f"Warning: Could not find layer {name} in ViT-B-16 LAION CLIP model")

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


def load_vit_b16_laion_clip_model(model_name: str | None = None, pretrained: str | None = None, device: torch.device | None = None):
    """Factory for ViT-B-16 LAION CLIP wrapper."""
    if model_name is None:
        model_name = ViTB16LAIONCLIPWrapper.DEFAULT_MODEL_NAME
    if pretrained is None:
        pretrained = ViTB16LAIONCLIPWrapper.DEFAULT_PRETRAINED

    print(f"Loading ViT-B-16 LAION CLIP model: {model_name} with {pretrained}")
    return ViTB16LAIONCLIPWrapper(model_name=model_name, pretrained=pretrained, device=device)


# Layer configuration for import by setup.py - all transformer blocks
VIT_B16_LAION_CLIP_LAYERS = [f"transformer.resblocks.{i}.mlp.c_fc" for i in range(12)]  # ViT-B-16: 12 blocks 