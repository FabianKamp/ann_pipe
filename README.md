# Activation Extraction Pipeline

## Overview
This repository packages the model setup, exposure, and activation extraction stack used to study orientation and working-memory representations with deep neural networks (DNNs). The codebase lets you initialize a curated panel of CNN, transformer, and vision-language models, expose them to controlled visual stimuli, and persist layer-wise activations to HDF5 for downstream representational similarity analysis (RSA) or decoding experiments.

## Installation
1. **Conda (recommended)**
   ```bash
   conda env create -f abstraction_perception.yaml
   conda activate abstraction_perception
   ```
2. **Pip-only (CPU)**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   GPU builds of PyTorch / torchvision should match your CUDA runtime if acceleration is needed.

## Repository Layout
```
ann_pipe/
├── abstraction_perception.yaml   # Full conda environment
├── requirements.txt              # Minimal pip dependencies
├── runners/
│   └── run_extract_activations.py
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py            # get_image_set helper
│   │   ├── lazy_activations.py   # memory-efficient HDF5 reading
│   │   └── preprocessing.py      # transforms + activation hooks
│   ├── models/                   # model configs + wrappers
│   └── utils/
│       └── abstr_perc_helperfuncs.py
├── examples/
│   └── run_single_model.py       # minimal extraction example
└── tests/
    └── validate_extraction.py    # output structure validation
```

## Quick Start
```bash
python runners/run_extract_activations.py \
    --stimuli_root /path/to/stimuli/semantic \
    --folders stimuli_rotatedsemantic \
    --output_dir /path/to/output/activations \
    --models ResNet50 CORNet-S ViT-B-16 \
    --replicates 4 --auto_confirm
```
Each folder listed under `--folders` produces an HDF5 file named `{output_prefix}_{folder}.h5`. The default prefix is `model_activations`.

## Core Components
- `src.models.setup`: Declares available architectures, their extraction layers, and loader helpers. The wrappers expose logits + intermediate activations in a consistent dictionary.
- `src.data.preprocessing`: Shared preprocessing transforms (ImageNet, CLIP, SLIP), alpha masking, replicate jitter, and hook-based activation capture.
- `src.data.loaders.get_image_set`: Lists `.png` stimuli within a folder.
- `src.data.lazy_activations`: Memory-efficient lazy-loading utilities for reading extracted activations from HDF5 files.
- `runners/run_extract_activations.py`: CLI orchestration for model initialization, image exposure, replicate averaging, and HDF5 persistence.

## Usage Patterns
### Multiple Folders
```bash
python runners/run_extract_activations.py \
    --stimuli_root ./stimuli/concrete \
    --folders stimuli_rotatedconcrete stimuli_confound/leaves \
    --output_dir ./output/concrete/activations \
    --output_prefix concrete_models
```
### Full Model Panel
Omit `--models` to extract from every model defined in `src/models/setup.py`.

### Working-Memory Oriented Experiments
- Use higher `--replicates` to stabilize representations under input jitter before computing working-memory RSA or bias metrics.
- Group output HDF5 files by manipulation (baseline vs. confound) to run depth-binned or layer-wise RDM comparisons downstream.

## CLI Reference
| Argument | Description |
| --- | --- |
| `--stimuli_root` | Base directory containing image folders. Required. |
| `--folders` | One or more subfolders (relative or absolute). Default: `.` |
| `--output_dir` | Destination directory for HDF5 files. Required. |
| `--output_prefix` | Filename prefix, defaults to `model_activations`. |
| `--models` | Subset of model names; default uses all configured models. |
| `--replicates` | Number of jittered exposures per image (default 4). |
| `--auto_confirm` | Skip the interactive safety prompt. |

## Output Data Format
- File structure: `/{model_name}/{stimulus_id}/{layer_name}`
- Dataset shape: `(n_replicates, feature_dim)`
- `stimulus_id` corresponds to the base filename (sans extension).
- To verify integrity:
  ```python
  import h5py
  with h5py.File("model_activations_rotatedsemantic.h5") as f:
      print(f["ResNet50"].keys())                # image ids
      acts = f["ResNet50"]["object_01"]["layer1.0.conv1"][:]
      assert acts.shape[0] == 4
  ```
The layout ensures compatibility with lazy-loading utilities or streaming RDM builders used in downstream RSA / working-memory decoding suites.

## Reading Extracted Activations
The pipeline includes memory-efficient lazy-loading utilities for reading HDF5 activation files without loading everything into RAM at once.

### Basic Usage
```python
from src.data.lazy_activations import load_lazy_activations

# Load activations with lazy loading
acts = load_lazy_activations("model_activations_semantic.h5")

# Access like a regular nested dictionary - data loads only when accessed
resnet_layer1 = acts["ResNet50"]["object_01"]["layer1.0.conv1"]  # (n_replicates, features)
vit_layer = acts["ViT-B-16"]["object_02"]["blocks.5.attn"]

# Works with loops - memory efficient for large datasets
for img_id in acts["ResNet50"].keys():
    for layer_name in acts["ResNet50"][img_id].keys():
        activation = acts["ResNet50"][img_id][layer_name]  # Loaded on demand
        # ... process activation
```

### Inspection Utilities
```python
from src.data.lazy_activations import (
    get_available_models,
    get_available_images,
    get_available_layers,
    get_activation_shape
)

# Inspect HDF5 structure without loading data
models = get_available_models("activations.h5")
# ['ResNet50', 'CORNet-S', 'ViT-B-16']

images = get_available_images("activations.h5", "ResNet50")
# ['object_01', 'object_02', ...]

layers = get_available_layers("activations.h5", "ResNet50")
# ['layer1.0.conv1', 'layer2.0.conv1', ...]

shape = get_activation_shape("activations.h5", "ResNet50")
# (4, 64)  -> (n_replicates, feature_dim)
```

### Memory Management
```python
# Lazy loading automatically caches small activations (< 10MB)
# Clear cache to free memory when needed
acts["ResNet50"]["object_01"].clear_cache()  # Clear single image
acts["ResNet50"].clear_cache()               # Clear entire model
acts.close()                                 # Clear all caches
```

### Single Model Loading
```python
from src.data.lazy_activations import load_model_activations_lazy

# Load only specific model
resnet_acts = load_model_activations_lazy("activations.h5", "ResNet50")
layer_data = resnet_acts["ResNet50"]["object_01"]["layer4.2.conv3"]
```

## Supported Models
| Training Type | CNN-based | Transformer-based |
| --- | --- | --- |
| Image Only | ResNet50, ResNeXt-101-WSL, ConvNeXt-Large, CORNet-S, VGG19 | ViT-B-16, ViT-B-32-timm, DeiT-Base |
| Image + Text | ResNet50-CLIP, ConvNeXt-Large-CLIP | ViT-B-16-CLIP, ViT-B-32-CLIP, ViT-B-16-LAION-CLIP |
| Contrastive | SLIP ViT-Small | — |

Add new models by extending `src/models/setup.py` with layer lists and loader factories.

## Troubleshooting & Performance Tips
- **Device placement**: The runner auto-selects MPS → CUDA → CPU. Override by editing `process_dataset` if you need explicit devices per model.
- **Memory pressure**: Use `--models` to process a subset or run multiple passes, appending into an existing HDF5 (the script opens files in append mode when partial subsets are requested).
- **Throughput**: Increase `--replicates` only when necessary; each replicate re-processes the full model forward pass.
- **Missing weights**: Some wrappers (e.g., SLIP, open_clip) need manual weight downloads; see inline error messages.

## Interoperation Notes
- Output HDF5 files feed directly into the RSA / bias analysis scripts (`runners_clean/run_analysis.py` in the original project). Copy the generated files into `output/activations/<orientation>/` to reuse existing RDM builders.
- Maintain a consistent naming convention (`output_prefix`) across experimental conditions to simplify behavioral alignment and depth-binned permutation tests.

## Examples
- `examples/run_single_model.py`: Minimal extraction workflow for a single model
- `examples/read_activations_example.py`: Demonstrates reading and inspecting extracted HDF5 files
  ```bash
  python examples/read_activations_example.py model_activations_semantic.h5 --model ResNet50 --show_layers
  ```

## Validation & Testing
- Run `tests/validate_extraction.py` to verify activation extraction output structure on a small stimulus set.
- Use `examples/read_activations_example.py` to inspect generated HDF5 files and verify layer shapes.
- For integration with neuroscience data, verify that the averaged replicates maintain the expected contrast (e.g., orientation tuning curves) before computing metrics.
