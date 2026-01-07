#%% Imports
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
import os
from PIL import Image
from scipy.stats import pearsonr
from tqdm import tqdm
from . import WM_RSA as WM
from . import memfMRI_funcs as memf
from scipy.spatial.distance import pdist, squareform
import random
import matplotlib.pyplot as plt
import seaborn as sns
import rsatoolbox.rdm as rsardm  # Import with a different name
import rsatoolbox.data as rsadata
import rsatoolbox.inference as rsainf
import cmath
import h5py

#test


#%% Functions
def circular_distance(p, q, angular_space):
    """
    Calculate arc-length (circular distance) between two points along a periodic variable.
    This function computes the circular distance between two sets of points, `p` and `q`, 
    which are defined along a periodic variable. The periodic nature of the variable is 
    taken into account using a specified boundary value, `angular_space`, at which the 
    variable wraps back to zero. This is particularly useful for angles where, for example, 
    360 degrees is equivalent to 0 degrees.
    Parameters
    ----------
    p : np.array | shape (n,)
        Vector of points along the periodic variable.
        
    q : np.array | shape (n,)
        Vector of points along the periodic variable.
        
    angular_space : scalar
        Periodic boundary (value at which the variable wraps back to zero), such as 360 for degrees 
        or 2π for radians.
    Returns
    -------
    d : np.array
        An array containing the distances between each individual pair of points in `p` and `q`.
        
    How It Works
    ------------
    1. **Difference Calculation**: The function calculates the difference between each pair of points 
    in `p` and `q`, adjusting by adding half of the angular space. This helps in handling wrap-around cases.
    
    2. **Modulo Operation**: It then takes this difference modulo `angular_space`, ensuring that all 
    distances are within the specified range.
    
    3. **Final Adjustment**: Finally, subtracting `angular_space/2` centers the results around zero, 
    allowing for small values when points are close together and larger values when they are on opposite sides.
    Example Usage
    --------------
    >>> p = np.array([10, 350, 180])
    >>> q = np.array([350, 10, 0])
    >>> angular_space = 360
    >>> distances = circular_distance(p, q, angular_space)
    """
    
    d = (p - q + angular_space / 2) % angular_space - angular_space / 2
    return d

def circ_rsa_model(angles, angular_space):
    """
    Create a circular RSA model based on input angles.
    
    Parameters:
    angles (list): List of angles in degrees
    angular_space (scalar): Angular space of the circular variable (360 or 180)

    Returns:
    numpy.ndarray: RSA matrix with values normalized between 0 and 2
    """
    
    # Convert angles to numpy array
    angles_array = np.array(angles)[...,np.newaxis]

    # Calculate pairwise distances using the custom function
    distances = np.abs(squareform(pdist(angles_array, metric=circular_distance, angular_space=angular_space)))

    # Normalize relative to maximum possible distance in the angular space
    max_possible_distance = angular_space/2  # Maximum possible distance in the given angular space
    rdm = distances / max_possible_distance * 2  # Scale to match 0-2 range of attraction/repulsion models

    return rdm


def circ_bias_model(angles, bias_type, angular_space=360):
    """
    Create an RDM based on quadrant relationships for 360 or 180 degree spaces.
    
    Parameters:
    -----------
    angles : numpy.ndarray
        Array of angles in degrees
    bias_type : str
        'repulsion' or 'attraction'
    angular_space : int
        360 or 180, specifying the angular space
    
    Returns:
    --------
    numpy.ndarray
        16x16 Quadrant-based RDM with values normalized between 0 and 2
    """
    n = len(angles)
    rdm = np.zeros((n, n))
    
    if angular_space == 360:
        if bias_type == 'repulsion':
            quadrants = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3])
        elif bias_type == 'attraction':
            quadrants = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 0, 0])
    elif angular_space == 180:
        if bias_type == 'repulsion':
            quadrants = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1])
        elif bias_type == 'attraction':
            quadrants = np.array([0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0])
    else:
        raise ValueError("angular_space must be either 360 or 180")
    
    for i in range(n):
        for j in range(n):
            q1, q2 = quadrants[i], quadrants[j]
            
            if q1 == q2:
                rdm[i, j] = 0  # Same quadrant
            elif angular_space == 360 and abs(q1 - q2) == 2:
                rdm[i, j] = 2  # Opposite quadrant (only for 360-degree space)
            else:
                rdm[i, j] = 1  # Adjacent quadrant (and opposite for 180-degree space)
    
    # For 180° space, rescale the RDM to use the full 0-2 range to match 360° models
    if angular_space == 180:
        # Find the maximum value in the 180° RDM (should be 1)
        max_val = np.max(rdm)
        if max_val > 0:  # Avoid division by zero
            # Scale so that max value becomes 2
            rdm = rdm * (2.0 / max_val)
    
    return rdm


def normalize_image(image_tensor):
    """
    Normalize an image tensor.
    
    :param image_tensor: Input image tensor
    :return: Normalized image tensor
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return normalize(image_tensor)

# Function to extract activations
def extract_activations(*, image, model, device, layer_names=None, model_type='cornet', offset_x=0, offset_y=0):
    """
    Extract activations from specified layers of a neural network model.
    
    Parameters:
    -----------
    image : torch.Tensor
        Input image tensor
    model : torch.nn.Module
        Neural network model
    device : torch.device
        Device to run the model on
    layer_names : list, optional
        List of layer names to extract. If None, uses default layers for each model type
    model_type : str
        Type of model ('cornet', 'vgg19', 'alexnet')
    offset_x : int
        Horizontal offset for image
    offset_y : int
        Vertical offset for image
    
    Returns:
    --------
    dict : Dictionary of layer activations
    """
    activations = {}
    hooks = []

    def hook(name):
        def fn(_, __, output):
            if isinstance(output, tuple):
                activations[name] = output[0].detach().cpu().numpy()
            else:
                activations[name] = output.detach().cpu().numpy()
        return fn

    # Set up layer extraction based on model type
    if model_type == "cornet":
        if layer_names is None:
            layer_names = ["V1", "V2", "V4", "IT"]
        for name in layer_names:
            # For CORNet models, we need to handle the module structure correctly
            try:
                layer = getattr(model, name)
            except AttributeError:
                layer = getattr(model.module, name)
            hooks.append(layer.register_forward_hook(hook(name)))
    
    elif model_type == "alexnet":
        if layer_names is None:
            layer_names = ["features.2", "features.5", "features.7", "features.9", "features.12"]
        
        # Register hooks for specified layers
        for name in layer_names:
            layer = model
            for part in name.split('.'):
                layer = getattr(layer, part)
            hooks.append(layer.register_forward_hook(hook(name)))

    elif model_type == "vgg19":
        if layer_names is None:
            layer_names = ["maxpool1", "maxpool2", "maxpool3", "maxpool4", "maxpool5"]
        
        pool_count = 0
        for name, layer in model.features.named_children():
            if isinstance(layer, torch.nn.MaxPool2d):
                pool_count += 1
                pool_name = f"maxpool{pool_count}"
                if pool_name in layer_names:
                    hooks.append(layer.register_forward_hook(hook(pool_name)))

    
    else:
        raise ValueError(f"Unsupported model type: {model_type}")

    # Apply offset to the image
    offset_image_tensor = offset_image(image, offset_x, offset_y)

    # Forward pass
    with torch.no_grad():
        _ = model(offset_image_tensor.unsqueeze(0).to(device))

    # Remove hooks
    for h in hooks:
        h.remove()

    return activations

def offset_image(image_tensor, offset_x=0, offset_y=0):
    """
    Offset an image tensor by a given value in width (x) or height (y),
    maintaining the original dimensions. Wraps the image around the edges.
    
    Parameters:
    image_tensor (torch.Tensor): Input image tensor of shape (C, H, W)
    offset_x (float): Horizontal offset (positive values move right, negative left)
    offset_y (float): Vertical offset (positive values move down, negative up)
    
    Returns:
    torch.Tensor: Offset image tensor of the same shape as input
    """
    channels, height, width = image_tensor.shape
    assert channels == 3, "Input tensor must have 3 channels"
    
    # Convert offsets to integers
    offset_x = int(round(offset_x))
    offset_y = int(round(offset_y))
    
    # Create a new tensor with the same content as the input
    offset_tensor = image_tensor.clone()

    # Apply horizontal offset
    if offset_x != 0:
        offset_x = offset_x % width  # Ensure offset is within image width
        offset_tensor[:, :, offset_x:] = image_tensor[:, :, :-offset_x]
        offset_tensor[:, :, :offset_x] = image_tensor[:, :, -offset_x:]

    # Apply vertical offset
    if offset_y != 0:
        offset_y = offset_y % height  # Ensure offset is within image height
        offset_tensor[:, offset_y:, :] = image_tensor[:, :-offset_y, :]
        offset_tensor[:, :offset_y, :] = image_tensor[:, -offset_y:, :]

    return offset_tensor

def rotate_images(original_images, angles):
    # Initialize an empty list to hold all rotated images
    all_rotated_images = []

    for image in original_images:
        # Ensure the image has 3 channels (convert grayscale to RGB if needed)
        if image.dim() == 2 or (image.dim() == 3 and image.size(0) == 1):
            image = image.repeat(3, 1, 1) if image.dim() == 3 else image.unsqueeze(0).repeat(3, 1, 1)

        for angle in angles:
            # Rotate the image and add to the list
            rotated_image = transforms.functional.rotate(image, angle)
            all_rotated_images.append(rotated_image)

    # Convert the flat list of rotated images to a single tensor
    rotated_images_tensor = torch.stack(all_rotated_images)

    return rotated_images_tensor

def tensor_to_pil(tensor):
    """
    Convert a tensor to a PIL image.
    
    Parameters:
    tensor (torch.Tensor): Input tensor of shape (C, H, W)
    
    Returns:
    PIL.Image.Image: PIL image
    """
    return transforms.ToPILImage()(tensor)

def save_rdms(set_wibw, layer_idx, rdm_dict):
    """
    Extract RDMs and save them to a dictionary of numpy arrays.
    
    :param set_wibw: Dictionary containing RDMs
    :param layer_idx: Index of the layer to extract
    :param rdm_dict: Dictionary to store the extracted RDMs
    :return: Updated rdm_dict
    """
    keys = ['both', 'wi', 'bw']
    
    for key in keys:
        rdm = set_wibw[key][layer_idx]
        if isinstance(rdm, np.ndarray):
            rdm_matrix = rdm
        else:
            rdm_vector = rdm.get_vectors().flatten()
            rdm_matrix = WM.trans_vec_mat(rdm_vector, ncon=16)
        
        # Ensure the matrix is 16x16
        if rdm_matrix.shape != (16, 16):
            print(f"Warning: RDM shape is {rdm_matrix.shape}, expected (16, 16). Reshaping...")
            rdm_matrix = rdm_matrix.reshape(16, 16)
        
        # Save the RDM matrix to the dictionary
        if key not in rdm_dict:
            rdm_dict[key] = {}
        if layer_idx not in rdm_dict[key]:
            rdm_dict[key][layer_idx] = []
        rdm_dict[key][layer_idx].append(rdm_matrix)
    
    return rdm_dict

def plot_rdms(rdm_dict, layer_idx, set_idx=None):
    """
    Plot 16x16 RDMs for 'both', 'wi', and 'bw' cases.
    
    :param rdm_dict: Dictionary containing RDM matrices
    :param layer_idx: Index of the layer to plot
    :param set_idx: Index of the set to plot (if None, plot average across sets)
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    titles = ['Both', 'Within', 'Between']
    keys = ['both', 'wi', 'bw']
    
    for idx, (title, key) in enumerate(zip(titles, keys)):
        if set_idx is None:
            # Plot average RDM across sets
            rdm_matrix = np.mean(rdm_dict[key][layer_idx], axis=0)
        else:
            # Plot RDM for specific set
            rdm_matrix = rdm_dict[key][layer_idx][set_idx]
        
        sns.heatmap(rdm_matrix, ax=axes[idx], cmap='viridis', square=True)
        axes[idx].set_title(f'{title} RDM - Layer {layers_to_extract[layer_idx]}')
        axes[idx].set_xlabel('Orientation')
        axes[idx].set_ylabel('Orientation')
    
    if set_idx is None:
        plt.suptitle(f'Average RDMs - Layer {layers_to_extract[layer_idx]}', fontsize=16)
    else:
        plt.suptitle(f'RDMs - Layer {layers_to_extract[layer_idx]}, Set {set_idx + 1}', fontsize=16)
    
    plt.tight_layout()
    plt.show()

def save_rdm_dict_to_disk(rdm_dict, save_path):
    """
    Save the RDM dictionary to disk.
    
    :param rdm_dict: Dictionary containing RDM matrices
    :param save_path: Path to save the RDM dictionary
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    file_path = os.path.join(save_path, 'cornetS_layers_rdm_dict.npz')
    
    # Convert the nested dictionary to a format that can be saved with np.savez
    save_dict = {}
    for condition in rdm_dict:
        for layer_idx in rdm_dict[condition]:
            key = f"{condition}_layer{layer_idx}"
            save_dict[key] = np.array(rdm_dict[condition][layer_idx])
    
    np.savez(file_path, **save_dict)
    print(f"RDM dictionary saved to {file_path}")

def normalize_and_count_zeros(image_sets, image_folder):
    """
    Normalize all images and count the number of zero values per image.
    
    :param image_sets: List of image sets (set_1, set_2, set_3)
    :param image_folder: Path to the folder containing images
    :return: Dictionary with image names as keys and number of zero values as values
    """
    zero_counts = {}
    
    for image_set in image_sets:
        for image_name in image_set:
            for ort in range(1, n_ort + 1):
                img_path = os.path.join(image_folder, f'{image_name}_{ort}.png')
                
                # Load and preprocess the image
                image = load_and_preprocess_image(img_path)
                
                # Normalize the image
                normalized_image = normalize_image(image)
                
                # Count zero values
                zero_count = torch.sum(normalized_image == 0).item()
                
                # Store the result
                zero_counts[f'{image_name}_{ort}.png'] = zero_count
    
    return zero_counts

def process_and_rotate_images(normalized_images, angles):
    """
    Rotate normalized images and adjust zero values.
    
    :param normalized_images: List of tuples (image_name, normalized_image_tensor)
    :param angles: List of angles to rotate the images
    :return: List of tuples (image_name, original_image, rotated_image)
    """
    processed_images = []
    
    for image_name, normalized_image in normalized_images:
        # Record the most common value (excluding exact zeros)
        most_common = torch.mode(normalized_image[normalized_image != 0]).values.item()
        
        for angle in angles:
            # Rotate the image
            rotated_image = rotate_images([normalized_image], [angle])[0]
            
            # Replace zero values with the most common value
            rotated_image[rotated_image == 0] = most_common
            
            processed_images.append((f"{image_name}_{angle}", normalized_image, rotated_image))
    
    return processed_images

def plot_images(original, processed, image_name, rotation_angle):
    """
    Plot original and processed images side by side.
    
    :param original: Original normalized image tensor or None
    :param processed: Processed (rotated and adjusted) image tensor or None
    :param image_name: Name of the image
    :param rotation_angle: Angle of rotation applied
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Handle original image
    if original is not None:
        original_np = original.permute(1, 2, 0).numpy()
        ax1.imshow(original_np)
        ax1.set_title('Original Normalized Image')
    ax1.axis('off')
    
    # Handle processed image
    if processed is not None:
        processed_np = processed.permute(1, 2, 0).numpy()
        ax2.imshow(processed_np)
        ax2.set_title(f'Rotated ({rotation_angle}°) and Adjusted Image')
    ax2.axis('off')
    
    plt.suptitle(f"Image: {image_name}")
    plt.tight_layout()
    plt.show()

def create_offset_models(n_conditions):
    """
    Create model RDMs specifically for the offset analysis with the correct number of conditions.
    
    :param n_conditions: Number of conditions in the offset analysis
    :return: Dictionary containing the model RDMs
    """
    # Create angles for each condition
    angles = np.linspace(0, 360, n_ort, endpoint=False)  # Using n_ort from global scope
    
    # Create the full angle list that matches the data structure
    n_images = n_conditions // n_ort
    full_angles = np.tile(angles, n_images)
    
    # Create 360° model RDM
    model_360 = circ_rsa_model(full_angles, angular_space=360)
    
    # Create 180° model RDM
    model_180 = circ_rsa_model(full_angles, angular_space=180)
    
    # Create model specifications
    models360 = {'angular': WM.create_model_spec_RSA_TB(rdm_array=np.array([model_360]), model_name='angular_360')}
    models180 = WM.create_model_spec_RSA_TB(rdm_array=np.array([model_180]), model_name='angular_180')
    
    return models360, models180

# First, let's add a function to calculate the number of conditions in the offset analysis
def get_offset_conditions(offset_images, angles):
    """
    Calculate the total number of conditions for the offset analysis.
    
    :param offset_images: List of (name, image) tuples
    :param angles: List of rotation angles
    :return: Total number of conditions
    """
    return len(offset_images) * len(angles)

def perform_offset_analysis(offset_layer_stat_mat, n_conditions, layers_to_extract, models360, models180, models_img):
    """
    Perform RSA analysis on offset data.
    
    :param offset_layer_stat_mat: List of layer activations for offset images
    :param n_conditions: Number of conditions in the offset analysis
    :param layers_to_extract: List of layer names
    :param models360: 360-degree model dictionary
    :param models180: 180-degree model dictionary
    :param models_img: Image model dictionary
    :return: Dictionary containing results
    """
    n_layers = len(layers_to_extract)
    
    # Perform RSA
    offset_set_rsa = WM.layer_RDMS_RSA(offset_layer_stat_mat, ncon=n_conditions, method=dist_method, layers=n_layers)
    offset_set_wibw = WM.sep_RDM(offset_set_rsa, ncon)
    
    # Initialize results dictionary
    results = {
        'both_360': WM.model_eval_first_level(offset_set_wibw['both'], model=models360['angular'], n_layers=n_layers, compare_method=compare_method),
        'wi_360': WM.model_eval_first_level(offset_set_wibw['wi'], model=models360['angular'], n_layers=n_layers, compare_method=compare_method),
        'bw_360': WM.model_eval_first_level(offset_set_wibw['bw'], model=models360['angular'], n_layers=n_layers, compare_method=compare_method),
        'both_180': WM.model_eval_first_level(offset_set_wibw['both'], model=models180, n_layers=n_layers, compare_method=compare_method),
        'wi_180': WM.model_eval_first_level(offset_set_wibw['wi'], model=models180, n_layers=n_layers, compare_method=compare_method),
        'bw_180': WM.model_eval_first_level(offset_set_wibw['bw'], model=models180, n_layers=n_layers, compare_method=compare_method)
    }
    
    # Image encoding analysis
    ncon_img_offset = min(9, n_conditions // 3)
    offset_set_rsa_img_full = WM.cv_split_img(offset_layer_stat_mat, ncon=n_conditions, ncon_img=ncon_img_offset, n_layers=n_layers, dist_method=dist_method, iter=iter)
    results['img'] = WM.model_eval_first_level(offset_set_rsa_img_full, model=models_img['image_wi'], n_layers=n_layers, compare_method=compare_method)
    
    return results

def analyze_orientation_biases(layer_stat_mat, layers, n_layers, ncon, dist_method, compare_method):
    """
    Analyze both 180° and 360° orientation biases for a given model
    """
    # Initialize arrays for both 360° and 180° results
    results = {
        "360": {
            "repulsion": np.zeros((n_layers, 1)),
            "attraction": np.zeros((n_layers, 1))
        },
        "180": {
            "repulsion": np.zeros((n_layers, 1)),
            "attraction": np.zeros((n_layers, 1))
        }
    }

    # Perform RSA analysis
    set_rsa = WM.layer_RDMS_RSA(layer_stat_mat, ncon=ncon, 
                               method=dist_method, layers=n_layers)
    set_wibw = WM.sep_RDM(set_rsa, ncon)

    # Calculate correlations for each layer
    for layer_idx in range(n_layers):
        layer_rdm = set_wibw["both"][layer_idx]
        
        # 360° correlations
        repulsion_corr_360 = WM.model_eval_first_level(
            [layer_rdm], 
            model=models360["repulsion"],
            n_layers=1, 
            compare_method=compare_method
        )
        attraction_corr_360 = WM.model_eval_first_level(
            [layer_rdm], 
            model=models360["attract"],
            n_layers=1, 
            compare_method=compare_method
        )
        
        # 180° correlations
        repulsion_corr_180 = WM.model_eval_first_level(
            [layer_rdm], 
            model=models180["repulsion"],
            n_layers=1, 
            compare_method=compare_method
        )
        attraction_corr_180 = WM.model_eval_first_level(
            [layer_rdm], 
            model=models180["attract"],
            n_layers=1, 
            compare_method=compare_method
        )
        
        # Store results
        results["360"]["repulsion"][layer_idx, 0] = repulsion_corr_360[0]
        results["360"]["attraction"][layer_idx, 0] = attraction_corr_360[0]
        results["180"]["repulsion"][layer_idx, 0] = repulsion_corr_180[0]
        results["180"]["attraction"][layer_idx, 0] = attraction_corr_180[0]
    
    return results

def plot_combined_biases(results, layers, model_name):
    """
    Plot combined 180° and 360° orientation biases
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 360° plot
    ax1.plot(range(len(layers)), results["360"]["repulsion"][:, 0], "o-", 
             label="Repulsion Bias", color="blue")
    ax1.plot(range(len(layers)), results["360"]["attraction"][:, 0], "o-", 
             label="Attraction Bias", color="red")
    ax1.set_xticks(range(len(layers)))
    ax1.set_xticklabels(layers)
    ax1.set_xlabel(f"{model_name} Layers")
    ax1.set_ylabel("Model Fit (CorrCov)")
    ax1.set_ylim(-1, 1)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()
    ax1.set_title("360° Orientation Bias")

    # 180° plot
    ax2.plot(range(len(layers)), results["180"]["repulsion"][:, 0], "o-", 
             label="Repulsion Bias", color="blue")
    ax2.plot(range(len(layers)), results["180"]["attraction"][:, 0], "o-", 
             label="Attraction Bias", color="red")
    ax2.set_xticks(range(len(layers)))
    ax2.set_xticklabels(layers)
    ax2.set_xlabel(f"{model_name} Layers")
    ax2.set_ylabel("Model Fit (CorrCov)")
    ax2.set_ylim(-1, 1)
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()
    ax2.set_title("180° Orientation Bias")

    plt.suptitle(f"Orientation Biases Across {model_name} Layers")
    plt.tight_layout()
    plt.show()

def permutation_test_layer(model_rdm, neural_rdm, n_permutations=1000):
    """
    Perform a permutation test comparing model and neural RDMs for a single layer.
    """
    # Convert ModelFixed to numpy array if needed
    if hasattr(model_rdm, "rdm_obj"):
        model_rdm = model_rdm.rdm_obj.get_matrices()[0]
    elif hasattr(model_rdm, "get_matrices"):
        model_rdm = model_rdm.get_matrices()[0]
    
    # Handle neural RDMs object if needed
    if hasattr(neural_rdm, "get_matrices"):
        neural_rdm = neural_rdm.get_matrices()[0]
    
    # Ensure RDMs are in matrix form
    if len(model_rdm.shape) == 1:
        model_rdm = WM.trans_vec_mat(model_rdm, int(np.sqrt(len(model_rdm) * 2)))
    
    # Ensure both RDMs have the same shape
    if model_rdm.shape != neural_rdm.shape:
        raise ValueError(f"RDM shapes don't match: model {model_rdm.shape} vs neural {neural_rdm.shape}")
    
    # Extract the upper triangular part (excluding diagonal)
    n = model_rdm.shape[0]
    model_flat = model_rdm[np.triu_indices(n, k=1)]
    neural_flat = neural_rdm[np.triu_indices(n, k=1)]
    
    # Calculate original correlation
    original_corr, _ = pearsonr(model_flat, neural_flat)
    
    # Perform permutations
    permuted_corrs = []
    for _ in tqdm(range(n_permutations), desc="Running permutations"):
        # Permute the rows and columns of the model RDM
        perm_indices = np.random.permutation(n)
        permuted_rdm = model_rdm[perm_indices][:, perm_indices]
        
        # Get permuted correlation
        perm_flat = permuted_rdm[np.triu_indices(n, k=1)]
        perm_corr, _ = pearsonr(perm_flat, neural_flat)
        permuted_corrs.append(perm_corr)
    
    # Calculate p-value
    p_value = np.mean(np.abs(permuted_corrs) >= np.abs(original_corr))
    
    return original_corr, p_value

def run_layer_permutation_tests(model_dict, neural_rdms, layers_to_test, n_permutations=1000):
    """
    Run permutation tests for each layer and model combination.
    
    Parameters:
    -----------
    model_dict : dict
        Dictionary containing model RDMs (ModelFixed objects or numpy arrays)
    neural_rdms : list
        List of neural RDMs, one per layer
    layers_to_test : list
        List of layer names to test
    n_permutations : int
        Number of permutations per test
    
    Returns:
    --------
    dict : Results containing correlations and p-values for each layer
    """
    results = {}
    
    for layer_idx, layer_name in enumerate(layers_to_test):
        print(f"\nTesting layer: {layer_name}")
        
        # Get neural RDM for this layer
        neural_rdm = neural_rdms[layer_idx]
        
        # Test each model
        layer_results = {}
        for model_name, model_rdm in model_dict.items():
            print(f"Testing model: {model_name}")
            
            corr, p_val = permutation_test_layer(
                model_rdm=model_rdm,
                neural_rdm=neural_rdm,
                n_permutations=n_permutations
            )
            
            layer_results[model_name] = {
                "correlation": corr,
                "p_value": p_val
            }
        
        results[layer_name] = layer_results
    
    return results

def compute_orientation_rdms(layer_stat_mat, n_orientations=16, dist_method='correlation'):
    """
    Compute RDMs based on orientations for each layer.
    
    Parameters:
    -----------
    layer_stat_mat : list
        List of layer activations, where each activation corresponds to an orientation
    n_orientations : int
        Number of orientations (default 16)
    dist_method : str
        Distance method for computing RDMs
        
    Returns:
    --------
    list : List of RDM objects, one per layer
    """
    n_layers = len(layer_stat_mat[0])
    layer_rdms = []
    
    # For each layer
    for layer_idx in range(n_layers):
        # Get activations for this layer
        layer_activations = []
        for i in range(n_orientations):
            # Get activation for this orientation
            orientation_activation = layer_stat_mat[i][layer_idx]
            layer_activations.append(orientation_activation)
            
        # Convert to numpy array
        layer_activations = np.array(layer_activations)
        
        # Create dataset
        measurements = WM.prepare_dataset_structure(n_orientations, layer_activations)
        rsa_dataset = WM.create_rsa_ds_object(measurements)
        
        # Calculate RDM
        layer_rdm = rsardm.calc_rdm(rsa_dataset, descriptor=None, method=dist_method)
        layer_rdms.append(layer_rdm)
    
    return layer_rdms

def analyze_model_significance(model_name, model_config, layer_stat_mat, n_permutations=1000, test_method='euclidean'):
    """
    Analyze the statistical significance of model fits across layers using specified test method.
    
    Parameters:
    -----------
    model_name : str
        Name of the model to analyze
    model_config : dict
        Configuration dictionary containing model, layers, and model_type
    layer_stat_mat : list
        Layer activation matrices 
    n_permutations : int
        Number of permutations for statistical testing
    test_method : str
        Method for computing neural RDMs (default 'euclidean')
    """
    n_orientations = 16
    layers_to_extract = model_config["layers"]
    
    # Compute orientation-based RDMs using specified test method
    print(f"Calculating neural RDMs using {test_method} distance...")
    neural_rdms = compute_orientation_rdms(
        layer_stat_mat,
        n_orientations=n_orientations, 
        dist_method=test_method
    )
    
    # Print shapes for debugging
    print("\nRDM shapes:")
    for i, rdm in enumerate(neural_rdms):
        if hasattr(rdm, "get_matrices"):
            shape = rdm.get_matrices()[0].shape
        else:
            shape = rdm.shape
        print(f"Layer {i}: {shape}")
    
    # Initialize results dictionary
    all_results = {}
    
    # Loop through all models
    for model_type in models360.keys():
        print(f"\nAnalyzing {model_type}...")
        
        # Convert model RDM to match orientation structure
        model_rdm = models360[model_type]
        if hasattr(model_rdm, "rdm_obj"):
            model_mat = model_rdm.rdm_obj.get_matrices()[0]
        elif hasattr(model_rdm, "get_matrices"):
            model_mat = model_rdm.get_matrices()[0]
        else:
            model_mat = model_rdm
            
        # Reshape model RDM if needed
        if model_mat.shape != (n_orientations, n_orientations):
            model_mat = model_mat[:n_orientations, :n_orientations]
        
        # Initialize results for this model type
        model_results = {
            'correlations': np.zeros(len(layers_to_extract)),
            'p_values': np.zeros(len(layers_to_extract))
        }
        
        # Test each layer
        for layer_idx, layer_name in enumerate(layers_to_extract):
            print(f"Testing layer: {layer_name}")
            
            corr, p_val = permutation_test_layer(
                model_rdm=model_mat,
                neural_rdm=neural_rdms[layer_idx],
                n_permutations=n_permutations
            )
            
            model_results['correlations'][layer_idx] = corr
            model_results['p_values'][layer_idx] = p_val
            
        all_results[model_type] = model_results
    
    # Print summary results
    print(f"\nResults summary for {model_name} using {test_method} distance:")
    for model_type, results in all_results.items():
        print(f"\n{model_type}:")
        for layer_idx, layer_name in enumerate(layers_to_extract):
            print(f"  {layer_name}:")
            print(f"    Correlation: {results['correlations'][layer_idx]:.3f}")
            print(f"    P-value: {results['p_values'][layer_idx]:.3f}")
    
    return all_results

def triu_vector_to_square(vector, n):
    """
    Convert upper triangular vector (without diagonal) to square symmetric matrix.
    
    Parameters:
    -----------
    vector : np.ndarray
        1D array of length n*(n-1)/2 containing upper triangular values
    n : int
        Size of the desired square matrix
        
    Returns:
    --------
    np.ndarray
        n x n symmetric matrix
    """
    # Create empty matrix
    matrix = np.zeros((n, n))
    
    # Fill upper triangle
    idx = np.triu_indices(n, k=1)
    matrix[idx] = vector
    
    # Fill lower triangle
    matrix = matrix + matrix.T
    
    return matrix

def load_activations(file_path):
    """
    Load activations from HDF5 file into a nested dictionary structure.
    
    Parameters:
    -----------
    file_path : str
        Path to HDF5 file containing activations
    
    Returns:
    --------
    dict
        Nested dictionary with structure:
        {model_name: {image_name_ort: {layer_name: activation_array}}}
    """
    activations = {}
    
    with h5py.File(file_path, "r") as f:
        # Iterate through models
        for model_name in f.keys():
            activations[model_name] = {}
            
            # Iterate through images
            for image_ort in f[model_name].keys():
                activations[model_name][image_ort] = {}
                
                # Iterate through layers
                for layer_name in f[model_name][image_ort].keys():
                    # Load activation array
                    act_array = f[model_name][image_ort][layer_name][:]
                    activations[model_name][image_ort][layer_name] = act_array
    
    return activations


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

# Helper function to convert numpy arrays for JSON serialization
def convert_for_json(obj):
    """Convert numpy arrays and other non-serializable objects for JSON saving."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj
