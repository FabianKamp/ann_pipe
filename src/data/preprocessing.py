"""
Functions for preprocessing images and extracting model activations.
"""
import torch
import torchvision.transforms as transforms
import h5py
import os
from PIL import Image
from tqdm import tqdm
from ..models import initialize_models
import numpy as np
from torchvision.transforms import functional as F
from PIL import ImageOps
import h5py
from torchvision.transforms import InterpolationMode
from torchvision.models import (
    ResNet50_Weights,
    ConvNeXt_Large_Weights,
    VGG19_Weights,
    ViT_B_16_Weights,
    ResNeXt101_32X8D_Weights as Rx101WSL_Weights,
)
import timm
from timm.data import resolve_data_config, create_transform

# --- constants used by every CLIP variant -----------------------------------
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def _clip_transform(image_size=224):
    """Exact OpenAI / open-clip inference-time preprocessing."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.Lambda(lambda img: img.convert("RGB")),      # enforce 3-channel input
        transforms.ToTensor(),
        transforms.Normalize(_CLIP_MEAN, _CLIP_STD),
    ])

def load_transform(model: str):
    """
    Return the preprocessing transform expected by `model`,
    *without* instantiating the network or downloading any weights.
    
    Args:
        model (str): Model name identifier
        
    Returns:
        transforms.Compose: The model's preprocessing pipeline
        
    Raises:
        ValueError: If model name is unknown
    """
    if model == "ResNet50" or model == "resnet50":
        return ResNet50_Weights.IMAGENET1K_V1.transforms()

    if model == "ConvNeXt-Large" or model == "convnext_large":
        return ConvNeXt_Large_Weights.IMAGENET1K_V1.transforms()

    if model == "CORNet-S" or model == "cornets" or model == "cornet":
        # CORNet-S was ImageNet-trained but ships without weight metadata.
        return transforms.Compose([
            transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    if model == "VGG19" or model == "vgg19" or model == "vgg19_wrapper":
        return VGG19_Weights.IMAGENET1K_V1.transforms()

    if model == "ViT-B-16" or model == "vit_b16" or model == "vit16":
        return ViT_B_16_Weights.IMAGENET1K_V1.transforms()

    if model == "ViT-B-32-timm" or model == "vit_b32_timm":
        cfg = timm.create_model("vit_base_patch32_224", pretrained=False).default_cfg
        return create_transform(**resolve_data_config(cfg))

    if model == "ResNeXt-101-WSL" or model == "resnext_wsl_wrapper":
        return Rx101WSL_Weights.DEFAULT.transforms()

    # ---------- CLIP families ----------------------------------------------
    if model in {
        "ResNet50-CLIP", "resnet50_clip", "ConvNeXt-Large-CLIP", "convnext_large_clip",
        "ViT-B-16-CLIP", "vit_b16_clip", "ViT-B-32-CLIP", "vit_b32_clip", "ViT-B-16-LAION-CLIP", "vit_b16_laion_clip"
    }:
        return _clip_transform(image_size=224)

    if model == "DeiT-Base" or model == "deit":
        cfg = timm.create_model("deit_base_patch16_224", pretrained=False).default_cfg
        return create_transform(**resolve_data_config(cfg))

    # ---------- CLIP-trained models (use CLIP normalization) --------------
    if model in {"resnet_clip", "convnext_clip", "vitclip", "vit_b32_clip", "vit_b16_laion_clip"}:
        return _clip_transform(image_size=224)


    raise ValueError(f"Unknown model name: {model}")

def transform_image(img_pil):
    """
    Apply standard preprocessing to a PIL image.
    """
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    return transform(img_pil)

def get_alpha_mask(image_path):
    """
    Extract a binary mask from the alpha channel of an RGBA image.
    Args:
        image_path (str): Path to image file
    Returns:
        np.ndarray: Binary mask (H, W), dtype bool
    """
    img = Image.open(image_path).convert("RGBA")
    alpha = np.array(img)[..., 3]
    return alpha > 0

def rgb_to_grayscale(rgb_img):
    """
    Convert an RGB image to grayscale using the standard luminance formula.
    Args:
        rgb_img (np.ndarray): RGB image (H, W, 3)
    Returns:
        np.ndarray: Grayscale image (H, W), dtype uint8
    """
    gray = np.dot(rgb_img[...,:3], [0.2989, 0.5870, 0.1140])
    return gray.astype(np.uint8)

def normalize_image(image_tensor):
    """
    Apply ImageNet normalization to image tensor.
    
    Args:
        image_tensor (torch.Tensor): Input image tensor
        
    Returns:
        torch.Tensor: Normalized image tensor
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    return normalize(image_tensor)

# -----------------------------------------------------------------------------
#  Input-jitter configuration
# -----------------------------------------------------------------------------
# A minimal random color jitter is applied for each replicate > 1.
# The small jitter is architecture-agnostic.
# E.g. 0.1 saturation factor: will be chosen uniformly from [0.9, 1.1] ( ±10% jitter).
JITTER_PARAMS = {
    "brightness": 0,
    "contrast": 0,
    "saturation": 0.05,
    "hue": 0.05,
}

def _build_jitter_transform(base_transform) -> transforms.Compose:
    """Return *base_transform* preceded by a small RandomAffine jitter."""

    jitter_tf = transforms.ColorJitter(**JITTER_PARAMS)
    
    # Handle both transforms.Compose and ImageClassification objects
    if hasattr(base_transform, 'transforms'):
        # This is a transforms.Compose object
        return transforms.Compose([jitter_tf] + list(base_transform.transforms))
    else:
        # This is an ImageClassification object (which is the actual transform)
        return transforms.Compose([jitter_tf, base_transform])

def preprocess_extract_activations(image_path, model, device, layers, model_type, n_replicates=1, replicate_index=0):
    """
    Preprocess image and extract activations from specified layers of a model.
    
    Args:
        image_path (str): Path to image file
        model (torch.nn.Module): Neural network model
        device (torch.device): Device to run model on
        layers (list): Names of layers to extract
        model_type (str): Type of model architecture
        n_replicates (int): Number of replicates for jittering (legacy, use replicate_index instead)
        replicate_index (int): Index of current replicate (0 = no jitter, >0 = apply jitter)
        
    Returns:
        dict: Layer activations {layer_name: activation_array}
    """
    # Get model device and ensure consistency
    model_device = next(model.parameters()).device
    if device is None or device != model_device:
        device = model_device

    # Load appropriate transform for model type
    transform = load_transform(model_type)
    
    # Apply jitter if this is not the first replicate (replicate_index > 0)
    if replicate_index > 0:
        transform = _build_jitter_transform(transform)

    activations = {}

    try:
        # Load and preprocess image
        with Image.open(image_path) as img:
            img = apply_alpha_mask(img) # remove alpha channel and output rgb
            img = pad_with_border(img, pad_ratio=0.1) # Apply padding to protect edges when resizing
            image_batch = transform(img).unsqueeze(0).to(device)
        
        # Extract activations for single replicate
        with torch.no_grad():
            if model_type in {"vitclip", "vit_b32_clip", "vit_b16_laion_clip", 
                             "resnet_clip", "convnext_clip", "vit16", "deit", "vit_b32_timm", "resnext_wsl_wrapper"}:
                # Models with built-in activation extraction
                _, features = model(image_batch)
                
                for layer_name in layers:
                    if layer_name in features:
                        feat = features[layer_name]
                        if len(feat.shape) == 3:  # (batch, tokens, features) - transformer
                            feat = feat.mean(dim=1)  # Average over tokens
                        activations[layer_name] = feat.cpu().numpy().reshape(1, -1)
                    else:
                        raise ValueError(f"Layer {layer_name} not found in {model_type} model activations")
            else:
                    # Use hooks for other models
                    hooks = []
                    temp_activations = {}
                    
                    def get_activation(name):
                        def hook(model, input, output):
                            if isinstance(output, tuple):
                                output = output[0]
                            temp_activations[name] = output.detach().cpu().numpy()
                        return hook
                    
                    # Check if model is wrapped in DataParallel
                    if hasattr(model, "module"):
                        base_model = model.module
                    else:
                        base_model = model
                    
                    # Register hooks based on model type
                    if model_type == "cornet":
                        for layer_name in layers:
                            layer = getattr(base_model, layer_name)
                            hooks.append(layer.register_forward_hook(get_activation(layer_name)))
                    elif model_type in ["vgg19", "vgg19_wrapper"]:
                        # VGG19 maxpool layers are at specific indices in the features module
                        vgg_layer_indices = {
                            "maxpool1": 4,    # First maxpool (after conv1_2)
                            "maxpool2": 9,    # Second maxpool (after conv2_2)
                            "maxpool3": 18,   # Third maxpool (after conv3_4)
                            "maxpool4": 27,   # Fourth maxpool (after conv4_4)
                            "maxpool5": 36    # Fifth maxpool (after conv5_4)
                        }
                        for layer_name in layers:
                            if layer_name in vgg_layer_indices:
                                layer_idx = vgg_layer_indices[layer_name]
                                if hasattr(base_model, 'model'):
                                    layer = base_model.model.features[layer_idx]
                                else:
                                    layer = base_model.features[layer_idx]
                                hooks.append(layer.register_forward_hook(get_activation(layer_name)))
                            else:
                                raise ValueError(f"Unknown VGG19 layer: {layer_name}")
                    else:
                        for layer_name in layers:
                            # Handle TimmModelWrapper which has the actual model as .model attribute
                            if hasattr(base_model, 'model') and hasattr(base_model, 'layer_names'):
                                # This is a TimmModelWrapper - use the inner model's named_modules
                                named_modules = dict([*base_model.model.named_modules()])
                            else:
                                # Regular model - use its named_modules directly
                                named_modules = dict([*base_model.named_modules()])
                                
                            if layer_name not in named_modules:
                                raise ValueError(f"Layer {layer_name} not found in model of type {model_type}")
                            layer = named_modules[layer_name]
                            hooks.append(layer.register_forward_hook(get_activation(layer_name)))
                    
                    # Forward pass with Apple Silicon fallback handling
                    try:
                        model(image_batch)
                    except RuntimeError as err:
                        msg = str(err)
                        # Apple-silicon quirk: certain conv shapes fall back to a slow CPU
                        # implementation. If that happens the input tensor gets moved to CPU
                        # implicitly while the weights stay on MPS → device mismatch.
                        if "slow_conv2d_forward_mps" in msg and "must be on the same device" in msg:
                            fallback_device = torch.device("cpu")
                            model.to(fallback_device)
                            image_batch = image_batch.to(fallback_device)
                            model(image_batch)
                            device = fallback_device
                        else:
                            raise
                    
                    # Process activations and flatten them
                    for layer_name in layers:
                        if layer_name in temp_activations:
                            act = temp_activations[layer_name].reshape(1, -1)  # Flatten
                            activations[layer_name] = act
                    
                    # Remove hooks
                    for hook in hooks:
                        hook.remove()
        
        # Validate that we got activations for all requested layers
        for layer_name in layers:
            if layer_name not in activations:
                raise ValueError(f"No activations collected for layer {layer_name}")
                
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        raise
    
    return activations

# -----------------------------------------------------------------------------

def extract_activations(image, model, device, layer_names, model_type, jitter=False):
    """
    Extract activations from specified layers of a model.
    
    Args:
        image (torch.Tensor): Input image tensor
        model (torch.nn.Module): Neural network model
        device (torch.device): Device to run model on
        layer_names (list): Names of layers to extract
        model_type (str): Type of model architecture
        jitter (bool): Whether to apply jittering
        
    Returns:
        dict: Layer activations {layer_name: activation_array}
    """
    # ------------------------------------------------------------------
    # Guarantee device consistency
    # ------------------------------------------------------------------
    model_device = next(model.parameters()).device
    if device is None or device != model_device:
        device = model_device

    activations = {}
    
    # Handle models with built-in activation extraction
    if model_type in {"slip", "vit", "vitclip", "vit16", "vit_b32_clip", "vit_b32_timm", 
                      "vit_b16_laion_clip", "resnext_wsl_wrapper", "deit", "swin", 
                      "levit", "resnet_clip", "convnext_clip", "convnext_large"}:
        with torch.no_grad():
            image_batch = image.unsqueeze(0).to(device)
            _, layer_activations = model(image_batch)
            
            # Process activations to match expected format
            for layer_name in layer_names:
                if layer_name in layer_activations:
                    # Mean over all tokens for transformer layers (if applicable)
                    act = layer_activations[layer_name]
                    if len(act.shape) == 3:  # (batch, tokens, features) - transformer
                        act = act.mean(dim=1)  # Average over tokens
                    activations[layer_name] = act.cpu().numpy()
                else:
                    raise ValueError(f"Layer {layer_name} not found in {model_type} model activations")
        return activations
    
    # For other models, use hooks
    hooks = []
    
    def get_activation(name):
        def hook(model, input, output):
            # Handle case where output is a tuple (e.g., for CORnet-RT)
            if isinstance(output, tuple):
                # Use first element of tuple
                output = output[0]
            activations[name] = output.detach().cpu().numpy()
        return hook
    
    # Check if model is wrapped in DataParallel
    if hasattr(model, "module"):
        base_model = model.module
    else:
        base_model = model
    
    # Register hooks based on model type
    if model_type == "cornet":
        for layer_name in layer_names:
            # Use the unwrapped model to get layers
            layer = getattr(base_model, layer_name)
            hooks.append(layer.register_forward_hook(get_activation(layer_name)))
    elif model_type in ["vgg19", "vgg19_wrapper"]:
        # VGG19 maxpool layers are at specific indices in the features module
        vgg_layer_indices = {
            "maxpool1": 4,    # First maxpool (after conv1_2)
            "maxpool2": 9,    # Second maxpool (after conv2_2)
            "maxpool3": 18,   # Third maxpool (after conv3_4)
            "maxpool4": 27,   # Fourth maxpool (after conv4_4)
            "maxpool5": 36    # Fifth maxpool (after conv5_4)
        }
        for layer_name in layer_names:
            if layer_name in vgg_layer_indices:
                layer_idx = vgg_layer_indices[layer_name]
                # For VGG19Wrapper, access the actual VGG model inside the wrapper
                if hasattr(base_model, 'model'):
                    layer = base_model.model.features[layer_idx]
                else:
                    layer = base_model.features[layer_idx]
                hooks.append(layer.register_forward_hook(get_activation(layer_name)))
            else:
                raise ValueError(f"Unknown VGG19 layer: {layer_name}")
    else:
        for layer_name in layer_names:
            # Get layer from named modules (handles nested layers better)
            named_modules = dict([*base_model.named_modules()])
            if layer_name not in named_modules:
                raise ValueError(f"Layer {layer_name} not found in model of type {model_type}")
            layer = named_modules[layer_name]
            hooks.append(layer.register_forward_hook(get_activation(layer_name)))
    
    # Forward pass
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        try:
            model(image_batch)
        except RuntimeError as err:
            msg = str(err)
            # ------------------------------------------------------------------
            # Apple-silicon quirk: certain conv shapes fall back to a slow CPU
            # implementation (``slow_conv2d_forward_mps``).  If that happens the
            # *input* tensor gets moved to CPU implicitly while the *weights*
            # stay on MPS → device mismatch.  We fall back to CPU for this
            # sample/model combination and retry once.
            # ------------------------------------------------------------------
            if "slow_conv2d_forward_mps" in msg and "must be on the same device" in msg:
                # Move both model & input to CPU and retry
                fallback_device = torch.device("cpu")
                model.to(fallback_device)
                image_batch = image_batch.to(fallback_device)
                model(image_batch)
                device = fallback_device  # Update for hook processing
            else:
                raise
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Flatten activations
    for layer_name in activations:
        activations[layer_name] = activations[layer_name].reshape(1, -1)
    
    return activations

def extract_and_save_activations(image_folder, output_path, image_set, n_orientations=16, selected_models=None):
    """
    Extract activations from multiple models and save to HDF5.
    
    Args:
        image_folder (str): Path to folder containing images
        output_path (str): Path to save HDF5 file
        image_set (list): List of image names
        n_orientations (int): Number of orientations per image
        selected_models (list, optional): List of model names to process. If None, processes all models.
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Prefer Apple Silicon GPU (mps) if available, otherwise CUDA, otherwise CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    # Initialize models
    models_config = initialize_models(device, selected_models=selected_models)
    
    # Determine mode - append if file exists and we're processing selected models, otherwise create new
    mode = "a" if os.path.exists(output_path) and selected_models is not None else "w"
    print(f"Opening HDF5 file in {mode} mode")
    
    # Create or open HDF5 file
    with h5py.File(output_path, mode) as f:
        # Create groups for each model
        for model_name in models_config.keys():
            if model_name not in f:
                print(f"Creating group for {model_name}")
                f.create_group(model_name)
        
        # Process each image
        for image_name in tqdm(image_set, desc="Processing images"):
            img_path = os.path.join(image_folder, f"{image_name}.png")
            
            try:
                # Process each model
                for model_name, config in models_config.items():
                    # Load and preprocess image with appropriate transform
                    transform = load_transform(model_name)
                    with Image.open(img_path).convert('RGB') as img:
                        image = transform(img)
                    
                    # Extract activations
                    activations = preprocess_extract_activations(
                        img_path,
                        config["model"],
                        device,
                        config["layers"],
                        config["model_type"]
                    )
                    
                    # Save activations for each layer
                    for layer_name in config["layers"]:
                        dataset_path = f"{model_name}/{image_name}/{layer_name}"
                        
                        # Delete existing dataset if it exists
                        if dataset_path in f:
                            del f[dataset_path]
                            
                        # Create new dataset
                        f.create_dataset(
                            dataset_path,
                            data=activations[layer_name]
                        )
                        
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")
                continue

# -----------------------------------------------------------------------------
# Low-level helper: apply alpha mask (for RGBA inputs)
# -----------------------------------------------------------------------------

def apply_alpha_mask(img_pil: Image.Image) -> Image.Image:
    """Return an *RGB* PIL image where transparent pixels have been zeroed.

    Parameters
    ----------
    img_pil : PIL.Image
        Input image.  Can be "RGB" or "RGBA"; if the latter, the alpha channel
        is used to mask out transparent regions.

    Returns
    -------
    PIL.Image.Image
        RGB image with transparent pixels set to black.
    """
    if img_pil.mode != "RGBA":
        # Ensure RGBA so we have an alpha channel (opaque default)
        img_pil = img_pil.convert("RGBA")

    img_np = np.array(img_pil)
    rgb = img_np[..., :3]
    alpha = img_np[..., 3] > 0  # boolean mask
    rgb[~alpha] = 0  # zero-out fully transparent pixels
    return Image.fromarray(rgb.astype(np.uint8), mode="RGB")

# -----------------------------------------------------------------------------
# Unified loader
# -----------------------------------------------------------------------------

def pad_with_border(img_pil: Image.Image, pad_ratio: float = 0.1, fill: tuple[int, int, int] = (0, 0, 0)) -> Image.Image:
    """Return a new PIL image with a constant border *pad_ratio* times the size.

    Parameters
    ----------
    img_pil : PIL.Image
        Input RGB image.
    pad_ratio : float, default 0.1
        Fraction of the corresponding width/height to pad on **each** side.
        A value of 0.1 means the new canvas will be 1.2× larger in each
        dimension (10 % border on every side).
    fill : tuple, default (0, 0, 0)
        RGB color for the border (defaults to black).
    """
    if pad_ratio <= 0:
        return img_pil

    w, h = img_pil.size
    pad_w = int(round(w * pad_ratio))
    pad_h = int(round(h * pad_ratio))
    return ImageOps.expand(img_pil, border=(pad_w, pad_h, pad_w, pad_h), fill=fill)

def load_and_preprocess_image(
    image_path: str,
    *,
    model_type: str | None = None,
    pre_pad_ratio: float = 0.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Load an image file and return a network-ready tensor.

    Parameters
    ----------
    image_path : str
        Path to the *PNG* file to load.
    model_type : {'slip', 'vit', ...}, optional
        If provided, a model-specific transform overrides the generic pipeline.
    pre_pad_ratio : float, default 0.0
        Fraction of the corresponding width/height to pad on **each** side before
        any resizing/cropping downstream. A value of 0.1 means the new canvas will
        be 1.2× larger in each dimension (10 % border on every side).
    device : torch.device, optional
        If set, the returned tensor is moved to that device.
    """

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    # ------------------
    # 1. Load & mask
    # ------------------
    with Image.open(image_path) as img:
        img_rgb = apply_alpha_mask(img)

    # Optional border padding to distance content from edges BEFORE any
    # resizing/cropping downstream.
    if pre_pad_ratio > 0:
        img_rgb = pad_with_border(img_rgb, pre_pad_ratio)

    # ------------------
    # 2. Choose transform
    transform = load_transform(model_type)
    tensor = transform(img_rgb)

    # ------------------
    # 3. Device move (optional)
    # ------------------
    if device is not None:
        tensor = tensor.to(device)

    return tensor

def transform_image_with_padding(img_pil, target_size=224):
    """
    Transform image while preserving aspect ratio and padding to square.
    This maintains the original aspect ratio and pads with zeros.
    
    Args:
        img_pil (PIL.Image): Input PIL image
        target_size (int): Target size for both dimensions
        
    Returns:
        torch.Tensor: Transformed and padded image tensor with shape [C, H, W]
    """
    # Get dimensions
    width, height = img_pil.size
    
    # Determine the longer dimension and scale factor
    longest_side = max(width, height)
    scale_factor = target_size / longest_side
    
    # Calculate new dimensions (preserving aspect ratio)
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    
    # Resize image preserving aspect ratio
    img_resized = img_pil.resize((new_width, new_height), Image.BILINEAR)
    
    # Calculate padding
    pad_left = (target_size - new_width) // 2
    pad_right = target_size - new_width - pad_left
    pad_top = (target_size - new_height) // 2
    pad_bottom = target_size - new_height - pad_top
    
    # Apply padding
    img_padded = F.pad(
        F.to_tensor(img_resized),
        [pad_left, pad_right, pad_top, pad_bottom],
        0  # padding value (black)
    )
    
    return img_padded

def process_images_with_padding(image_dir, file_pattern=".png"):
    """
    Process images with padding to preserve aspect ratio for analysis.
    
    Args:
        image_dir (str): Directory containing images
        file_pattern (str): File extension pattern to filter images
        
    Returns:
        tuple: (torch.Tensor of processed images, list of image filenames)
    """
    padded_images = []
    filenames = []
    
    # Get image files from the directory, filtering for image files
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(file_pattern)]
    total_images = len(image_files)
    
    print(f"Found {total_images} images to analyze")
    milestone = max(1, total_images // 10)  # Report progress at 10% intervals
    
    for i, image_name in enumerate(image_files):
        # Print progress at 10% intervals
        if (i % milestone == 0) or (i == total_images - 1):
            print(f"Processing image {i+1}/{total_images} ({(i+1)/total_images*100:.1f}%)")
            
        image_path = os.path.join(image_dir, image_name)
        try:
            img = Image.open(image_path).convert("RGBA")
            
            # Extract RGB and apply alpha masking
            img_np = np.array(img)
            rgb = img_np[..., :3]
            alpha = img_np[..., 3] > 0
            masked_rgb = rgb * alpha[..., None]
            
            # Convert back to PIL
            img_masked = Image.fromarray(masked_rgb.astype(np.uint8)).convert("RGB")
            
            # Apply padding transform
            img_tensor = transform_image_with_padding(img_masked)
            
            # Convert to grayscale
            img_gray = torch.mean(img_tensor, dim=0, keepdim=True)
            
            padded_images.append(img_gray)
            filenames.append(os.path.splitext(image_name)[0])  # Remove file extension
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
    
    if padded_images:
        return torch.stack(padded_images), filenames
    else:
        return None, []
