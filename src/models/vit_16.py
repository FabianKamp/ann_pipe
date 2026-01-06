"""
ViT-B/16 (HuggingFace) model loading and configuration.

Wrapper around `transformers.ViTForImageClassification` that exposes the same
activation-collection interface used elsewhere in the codebase.  The wrapper is
kept intentionally similar to `vit_clip.ViTWrapper` to preserve a consistent
API across Vision-Transformer flavours.
"""

from __future__ import annotations

import torch
from typing import Dict, List

try:
    # Transformers is an optional dependency for the project; we import lazily to
    # avoid hard failure if the user only needs other models.
    from transformers import ViTForImageClassification
except ModuleNotFoundError as err:  # pragma: no cover – runtime dependency guard
    raise ModuleNotFoundError(
        "`transformers` library is required to load ViT-B/16. Install via\n"
        "  pip install transformers>=4.40.0\n"
    ) from err


class ViT16Wrapper(torch.nn.Module):
    """Vision-Transformer (Base-16) wrapper exposing intermediate activations.

    The underlying model is `google/vit-base-patch16-224`.  We hijack the final
    feed-forward layer (often referred to as *Pre-LayerNorm FFN* or *pwff*) of a
    subset of blocks so that the resulting activation dictionary matches
    `setup.VIT_16_LAYERS`.
    """

    DEFAULT_CHECKPOINT: str = "google/vit-base-patch16-224"

    # Extract all transformer blocks (comprehensive layer extraction)
    LAYER_NAMES: List[str] = [f"transformer.blocks.{i}.pwff.fc2" for i in range(12)]  # ViT-Base: 12 blocks

    def __init__(self, checkpoint: str | None = None, *, device: torch.device | None = None):
        super().__init__()

        # --------------------------------------------------
        # Device resolution (CUDA > MPS > CPU)
        # --------------------------------------------------
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")

        if checkpoint is None:
            checkpoint = self.DEFAULT_CHECKPOINT

        # --------------------------------------------------
        # Model instantiation
        # --------------------------------------------------
        self.model = ViTForImageClassification.from_pretrained(checkpoint)
        self.model.to(device)
        self.model.eval()

        # --------------------------------------------------
        # Activation extraction bookkeeping
        # --------------------------------------------------
        self.activation: Dict[str, torch.Tensor] = {}
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []

        # Register hooks lazily so that activation dict is fresh for each forward
        self._register_hooks()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_submodule(self, root: torch.nn.Module, path: List[str]):
        """Resolve dotted path into a real sub-Module.

        Supports integer indexing into lists/`ModuleList`s *and* custom aliases
        to bridge naming differences between HuggingFace and timm conventions.
        """
        module: torch.nn.Module = root
        for part in path:
            if part.isdigit():
                module = module[int(part)]  # type: ignore[index]
            else:
                # Translate custom aliases first; some expand to multiple hops
                alias_map: dict[str, tuple[str, ...]] = {
                    "transformer": ("vit",),                 # Root encoder
                    "blocks": ("encoder", "layer"),         # ModuleList of ViTLayers
                    "pwff": ("output",),                    # FFN output part
                    "fc2": ("dense",),                      # Final linear layer
                }

                hops = alias_map.get(part, (part,))
                for hop in hops:
                    module = getattr(module, hop)
        return module

    def _register_hooks(self):
        """Attach forward hooks to pre-defined layers."""
        root = self.model
        for name in self.LAYER_NAMES:
            path_parts = name.split(".")
            layer = self._resolve_submodule(root, path_parts)
            handle = layer.register_forward_hook(self._make_hook(name))
            self._hook_handles.append(handle)

    def _make_hook(self, identifier: str):
        def hook_fn(_module, _inputs, output):
            self.activation[identifier] = output
        return hook_fn

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        """Return logits and collected activations."""
        logits = self.model(x).logits  # type: ignore[attr-defined]
        return logits, self.activation

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------
    def __del__(self):
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:  # pragma: no cover – best-effort cleanup
                pass


# ---------------------------------------------------------------------
# Public factory (mirrors vit_clip for backward compatibility)
# ---------------------------------------------------------------------

def load_vit_16_model(*, checkpoint: str | None = None, device: torch.device | None = None):
    """Factory for ViT-B/16 wrapper.

    Parameters
    ----------
    checkpoint: str | None, optional
        HuggingFace model identifier. Defaults to `google/vit-base-patch16-224`.
    device: torch.device | None, optional
        Device to instantiate the model on. If *None*, falls back to CUDA → MPS → CPU.
    """

    if checkpoint is None:
        checkpoint = ViT16Wrapper.DEFAULT_CHECKPOINT

    print(f"Loading ViT-B/16 model from HuggingFace: {checkpoint}")
    return ViT16Wrapper(checkpoint=checkpoint, device=device)


# Layer configuration for import by setup.py - all transformer blocks  
VIT_16_LAYERS = [f"transformer.blocks.{i}.pwff.fc2" for i in range(12)]  # ViT-Base: 12 blocks 