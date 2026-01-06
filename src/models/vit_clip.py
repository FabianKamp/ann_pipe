"""
ViT model loading and configuration.
Wrapper built on timm's ViT-Large CLIP implementation.
"""
import os
import torch
import timm
from typing import Dict

class ViTWrapper(torch.nn.Module):
    """Vision-Transformer wrapper exposing intermediate activations.

    This replaces the previous Hugging-Face ViT-B implementation with timm's
    `vit_large_patch14_clip_224.laion2b_ft_in1k` while preserving the forward
    interface `(logits, activation_dict)` expected by the rest of the codebase.
    """

    DEFAULT_MODEL_NAME = "vit_large_patch14_clip_224.laion2b_ft_in1k"

    def __init__(self, model_name: str | None = None, device: torch.device | None = None):
        super().__init__()

        # --------------------------------------------------
        # Device selection (CUDA > MPS > CPU)
        # --------------------------------------------------
        if device is None:
            if torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if model_name is None:
            model_name = self.DEFAULT_MODEL_NAME

        # --------------------------------------------------
        # Model loading
        # --------------------------------------------------
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.to(device)
        self.model.eval()

        # --------------------------------------------------
        # Activation extraction bookkeeping
        # --------------------------------------------------
        self.activation: Dict[str, torch.Tensor] = {}
        self.hook_handles: list[torch.utils.hooks.RemovableHandle] = []

        # Extract all transformer blocks (comprehensive layer extraction)
        self.layer_names = [f"blocks.{i}.mlp.fc2" for i in range(24)]  # ViT-Large has 24 blocks

        self._register_hooks()

    # --------------------------------------------------
    # Helpers
    # --------------------------------------------------
    def _get_submodule(self, path: str):
        """Resolve dotted access including integer indexing into ModuleLists."""
        module = self.model
        for part in path.split('.'):
            if part.isdigit():
                module = module[int(part)]  # type: ignore[index]
            else:
                module = getattr(module, part)
        return module

    def _register_hooks(self):
        for name in self.layer_names:
            layer = self._get_submodule(name)
            handle = layer.register_forward_hook(self._make_hook(name))
            self.hook_handles.append(handle)

    def _make_hook(self, name: str):
        def hook(_module, _input, output):
            self.activation[name] = output
        return hook

    # --------------------------------------------------
    # Forward
    # --------------------------------------------------
    def forward(self, x: torch.Tensor):
        """Return logits and collected activations."""
        logits = self.model(x)
        return logits, self.activation

    # --------------------------------------------------
    # Cleanup
    # --------------------------------------------------
    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()

# ---------------------------------------------------------------------
# Public factory (kept for backward compatibility)
# ---------------------------------------------------------------------

def load_vit_clip_model(model_name: str | None = None, device=None):
    """
    Backward-compatible alias to build the new ViT-Large-CLIP wrapper.
    """
    if model_name is None:
        model_name = ViTWrapper.DEFAULT_MODEL_NAME

    print(f"Loading vit_large_patch14_clip_224 model: {model_name}")
    return ViTWrapper(model_name, device)


# Layer configuration for import by setup.py - all transformer blocks
VIT_CLIP_LAYERS = [f"blocks.{i}.mlp.fc2" for i in range(24)]  # ViT-Large: 24 blocks 