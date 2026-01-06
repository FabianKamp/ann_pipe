"""
Script for extracting and saving neural network model activations.
"""
import argparse
import os
import sys
from typing import Iterable

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import setup  # noqa: E402
from src.data import preprocessing, loaders  # noqa: E402
import src.utils.abstr_perc_helperfuncs as aph  # noqa: E402
import torch
import h5py
from PIL import Image
import numpy as np
from torchvision import transforms


def _resolve_folder(stimuli_root: str, folder: str) -> str:
    if os.path.isabs(folder):
        return folder
    return os.path.join(stimuli_root, folder)


def main(
    *,
    stimuli_root: str,
    output_dir: str,
    folders: Iterable[str] = (".",),
    selected_models: Iterable[str] | None = None,
    n_replicates: int = 4,
    output_prefix: str = "model_activations",
    auto_confirm: bool = False,
):
    """
    Main function for extracting and saving model activations.
    
    Args:
        selected_models (list, optional): List of model names to process. If None, processes all models.
        n_replicates (int, optional): Number of input-jitter replicates per image.
        use_confounds (bool, optional): Whether to use confound stimuli or original stimuli (only applies to concrete type).
        stimuli_type (str, optional): Type of stimuli - "semantic" or "concrete".
    """
    stimuli_root = os.path.abspath(stimuli_root)
    output_dir = os.path.abspath(output_dir)
    folders = list(folders)

    if not folders:
        raise ValueError("At least one folder must be provided via --folders")

    os.makedirs(output_dir, exist_ok=True)
    print(f"Stimuli root: {stimuli_root}")
    print(f"Output directory: {output_dir}")
    
    # Display model selection info
    if selected_models:
        print(f"\nWill process only these models: {', '.join(selected_models)}")
    else:
        print("\nWill process all available models")
    
    # Process each folder
    total_images = 0
    folder_info = []
    
    for folder_name in folders:
        folder_path = _resolve_folder(stimuli_root, folder_name)
        display_name = os.path.basename(folder_path.rstrip(os.sep)) or "root"
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} not found, skipping...")
            continue
            
        # Get image set for this folder
        images = loaders.get_image_set(folder_path)
        if not images:
            print(f"Warning: No images found in {folder_path}, skipping...")
            continue
            
        folder_info.append((display_name, folder_path, images))
        total_images += len(images)
        print(f"Found {len(images)} images in {display_name}")
    
    if not folder_info:
        print("Error: No valid folders with images found!")
        return
    
    print(f"\nWill process a total of {total_images} images across {len(folder_info)} folders.")
    
    # Check if running in batch mode (no TTY available)
    if not auto_confirm and os.isatty(0):  # Interactive mode
        proceed = input("Proceed with activation extraction? (y/n): ")
        if proceed.lower() != 'y':
            print("Aborted by user.")
            return
    else:
        print("Auto-confirm enabled or running in batch mode – proceeding automatically.")
    
    # Process each folder
    for folder_name, folder_path, images in folder_info:
        print(f"\n{'='*60}")
        print(f"Processing folder: {folder_name}")
        print(f"{'='*60}")
        
        sanitized = folder_name.strip("./").replace(os.sep, "_") or "root"
        output_path = os.path.join(output_dir, f"{output_prefix}_{sanitized}.h5")
        
        process_dataset(
            f"{folder_name} images", 
            folder_path, 
            output_path, 
            images,
            selected_models,
            n_replicates=n_replicates,
        )
    
    print("\nActivation extraction complete!")
    for folder_name, _, _ in folder_info:
        sanitized = folder_name.strip("./").replace(os.sep, "_") or "root"
        output_path = os.path.join(output_dir, f"{output_prefix}_{sanitized}.h5")
        print(f"{folder_name} activations saved to: {output_path}")

def process_dataset(dataset_name, image_folder, output_path, image_set, selected_models=None, *, n_replicates: int = 1):
    """Process a single dataset, extracting activations for all models."""
    print(f"\nExtracting activations for {dataset_name}...")
    print(f"Saving to: {output_path}")
    
    # Set up device – prefer Apple Silicon GPU (mps), then CUDA, then CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")
    
    try:
        # Initialize models
        print("Initializing models...")
        models_config = setup.initialize_models(device, selected_models=selected_models)
        
        # Get list of models
        model_names = list(models_config.keys())
        print(f"Processing models: {', '.join(model_names)}")
        
        # Create or append to HDF5 file
        mode = "a" if os.path.exists(output_path) and selected_models is not None else "w"
        print(f"Opening HDF5 file in {mode} mode")
        
        with h5py.File(output_path, mode) as f:
            # Create groups for each model
            for model_name in models_config.keys():
                if model_name not in f:
                    print(f"Creating group for {model_name}")
                    f.create_group(model_name)
            
            # Process each model separately to handle errors gracefully
            for model_name, config in models_config.items():
                try:
                    print(f"\nProcessing model: {model_name}")
                    process_model(
                        h5_file=f,
                        model_name=model_name,
                        config=config,
                        image_folder=image_folder,
                        image_set=image_set,
                        device=device,
                        n_replicates=n_replicates,
                    )
                except Exception as e:
                    print(f"\nERROR processing model {model_name}: {str(e)}")
                    print("Continuing with other models...")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()

def process_model(
    h5_file,
    model_name: str,
    config,
    image_folder: str,
    image_set,
    device,
    *,
    n_replicates: int = 1,
):
    """Process a single model for all images."""
    from tqdm import tqdm
    
    model = config["model"]
    model_type = config["model_type"]
    layers = config["layers"]
    
    print(f"Model type: {model_type}")
    print(f"Extracting layers: {', '.join(layers)}")
    
    # Process each image
    for image_name in tqdm(image_set, desc=f"Processing {model_name}"):
        img_path = os.path.join(image_folder, f"{image_name}.png")
        
        try:
            # --------------------------------------------------------------
            #  Build replicate stack - each replicate gets different jittering
            # --------------------------------------------------------------
            replicate_accum = {ln: [] for ln in layers}

            # Process each replicate individually to ensure different jittering
            for rep_idx in range(n_replicates):
                # Get single replicate with proper jittering (first replicate has no jitter)
                acts = preprocessing.preprocess_extract_activations(
                    img_path,
                    model,
                    device,
                    layers,
                    model_type,
                    n_replicates=1,  # Single replicate per call
                    replicate_index=rep_idx  # Pass replicate index for jittering control
                )
                
                # Collect this replicate's activations
                for layer_name in layers:
                    replicate_accum[layer_name].append(acts[layer_name].squeeze(0))

            # --------------------------------------------------------------
            #  Save stacked replicates
            # --------------------------------------------------------------
            for layer_name, reps in replicate_accum.items():
                data = np.stack(reps, axis=0)  # (n_replicates, feats)
                dataset_path = f"{model_name}/{image_name}/{layer_name}"

                if dataset_path in h5_file:
                    del h5_file[dataset_path]

                h5_file.create_dataset(dataset_path, data=data)
                
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract model activations")
    parser.add_argument("--stimuli_root", required=True, help="Base directory containing stimulus folders (absolute or relative).")
    parser.add_argument("--folders", nargs="+", default=["."], help="One or more folders within the stimuli root to process (use '.' for the root).")
    parser.add_argument("--output_dir", required=True, help="Directory where activation HDF5 files will be stored.")
    parser.add_argument("--output_prefix", default="model_activations", help="Prefix for output HDF5 files (default: model_activations).")
    parser.add_argument("--models", nargs="+", default=None, help="Models to process (default: all available).")
    parser.add_argument("--replicates", type=int, default=4, help="Number of input-jitter replicates per image (default: 4).")
    parser.add_argument("--auto_confirm", action="store_true", help="Skip interactive confirmation prompt.")
    args = parser.parse_args()

    main(
        stimuli_root=args.stimuli_root,
        output_dir=args.output_dir,
        folders=args.folders,
        selected_models=args.models,
        n_replicates=args.replicates,
        output_prefix=args.output_prefix,
        auto_confirm=args.auto_confirm,
    )