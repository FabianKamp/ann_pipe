"""
ConvNeXt CLIP model loading and configuration.
Wrapper around open_clip's ConvNeXt implementation.
"""

import torch
from typing import Dict, List

try:
    import open_clip
except ModuleNotFoundError as err:
    raise ModuleNotFoundError(
        "`open_clip` library is required to load ConvNeXt CLIP. Install via\n"
        "  pip install open-clip-torch\n"
    ) from err


class ConvNeXtCLIPWrapper(torch.nn.Module):
    """ConvNeXt CLIP wrapper exposing intermediate activations."""

    DEFAULT_MODEL_NAME = "convnext_large_d_320"
    DEFAULT_PRETRAINED = "laion2b_s29b_b131k_ft_soup"

    # Layer names for activation extraction - all conv_dw layers from all stages and blocks
    LAYER_NAMES: List[str] = []
    
    # ConvNeXt-Large architecture: Stage 0(3 blocks), Stage 1(3 blocks), Stage 2(27 blocks), Stage 3(3 blocks)
    @classmethod
    def _generate_layer_names(cls):
        layer_names = []
        stage_blocks = [3, 3, 27, 3]  # blocks per stage
        for stage_idx, num_blocks in enumerate(stage_blocks):
            for block_idx in range(num_blocks):
                layer_names.append(f"trunk.stages.{stage_idx}.blocks.{block_idx}.conv_dw")
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
                print(f"Warning: Could not find layer {name} in ConvNeXt CLIP model")

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


def load_convnext_clip_model(model_name: str | None = None, pretrained: str | None = None, device: torch.device | None = None):
    """Factory for ConvNeXt CLIP wrapper."""
    if model_name is None:
        model_name = ConvNeXtCLIPWrapper.DEFAULT_MODEL_NAME
    if pretrained is None:
        pretrained = ConvNeXtCLIPWrapper.DEFAULT_PRETRAINED

    print(f"Loading ConvNeXt CLIP model: {model_name} with {pretrained}")
    return ConvNeXtCLIPWrapper(model_name=model_name, pretrained=pretrained, device=device)


# Layer configuration for import by setup.py - all conv_dw layers from all stages and blocks
CONVNEXT_CLIP_LAYERS = []
_stage_blocks = [3, 3, 27, 3]  # blocks per stage
for _stage_idx, _num_blocks in enumerate(_stage_blocks):
    for _block_idx in range(_num_blocks):
        CONVNEXT_CLIP_LAYERS.append(f"trunk.stages.{_stage_idx}.blocks.{_block_idx}.conv_dw") 