"""
SLIP model loading and configuration.
Based on Meta's SLIP implementation: https://github.com/facebookresearch/SLIP
"""
import os
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np

class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        self.proj = nn.Conv2d(
            in_chans, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # B, E, H//P, W//P
        x = x.flatten(2).transpose(1, 2)  # B, N, E
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class SLIPVisionTransformer(nn.Module):
    """Vision Transformer for SLIP, modified to expose intermediate activations."""
    
    def __init__(self, model_path, device=None):
        super().__init__()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
        # Model configuration (ViT-Small)
        self.img_size = 224
        self.patch_size = 16
        self.in_chans = 3
        self.embed_dim = 384  # ViT-Small
        self.depth = 12
        self.num_heads = 6
        self.mlp_ratio = 4.
        
        # Initialize architecture
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim
        )
        
        num_patches = self.patch_embed.n_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        self.blocks = nn.ModuleList([
            Block(self.embed_dim, self.num_heads, self.mlp_ratio)
            for _ in range(self.depth)
        ])
        
        # Load pretrained weights
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract vision transformer state dict
        state_dict = checkpoint["state_dict"]
        vision_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module.visual."):
                vision_state_dict[k.replace("module.visual.", "")] = v
        
        # Load state dict
        self.load_state_dict(vision_state_dict, strict=False)
        self.to(device)
        self.eval()
        
        # Register hooks for activation extraction
        self.activation = {}
        self.hook_handles = []
        
        def get_activation(name):
            def hook(model, input, output):
                self.activation[name] = output
            return hook
        
        # Register hooks for layers we want to extract
        for i, block in enumerate(self.blocks):
            if i in [1, 3, 6, 9, 11]:  # Layers we want to extract
                self.hook_handles.append(
                    block.register_forward_hook(
                        get_activation(f"blocks.{i}")))
    
    def forward(self, x):
        """Forward pass with activation extraction."""
        # Patch embedding
        x = self.patch_embed(x)
        
        # Add class token and position embedding
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Return both final output and intermediate activations
        return x, self.activation
    
    def __del__(self):
        """Clean up hooks when model is deleted."""
        for handle in self.hook_handles:
            handle.remove()

def load_slip_model(model_path=None, device=None):
    """
    Load SLIP model with pretrained weights.
    
    Args:
        model_path (str, optional): Path to model weights. If None, uses default path.
        device (torch.device, optional): Device to load model on.
        
    Returns:
        SLIPVisionTransformer: Loaded model
    """
    if model_path is None:
        # Get the project root directory (parent of src/)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(project_root, "models", "SLIP", "weights", "slip_small_100ep.pt")
        print(f"Project root: {project_root}")
        print(f"Looking for model at: {model_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model weights not found at {model_path}. "
            "Please download them from: "
            "https://dl.fbaipublicfiles.com/slip/slip_small_100ep.pt"
        )
    
    print(f"Found model weights at: {model_path}")
    return SLIPVisionTransformer(model_path, device) 