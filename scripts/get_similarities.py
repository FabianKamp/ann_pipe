import h5py
import argparse 
import os 
import numpy as np

results = []

def process(name, obj):
    if '/' not in name:
        print("=" * 80)
    indent = "  " * name.count('/')
    
    if isinstance(obj, h5py.Group):
        print(f"Processing {indent}Group: {name}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent}Dataset: {name} - shape: {obj.shape}, dtype: {obj.dtype}")
        get_pair_similarities(name, obj)

def get_pair_similarities(name, dataset):
    model, set_id, layer = name.split("/")
    data = dataset.value
    
    cor_mat = np.corrcoef(data, rowvar=True)

    lower_indices = np.tril_indices(n=cor_mat.shape[0], k=-1)    
    for i, j in zip(lower_indices[0], lower_indices[1]):
        results.append({
            "model": model,
            "set_id": set_id,
            "layer": layer,
            "img_a": i,
            "img_b": j,
            "correlation": cor_mat[i, j]
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recursively check all groups within h5 file')
    parser.add_argument('--file', type=str, required=True, help='Path to the h5 file')
    args = parser.parse_args()

    assert os.path.isfile(args.file), f"{args.file} not found. Check if path is correct."

    with h5py.File(args.file, 'r') as f:
        print("Top-level items:", list(f.keys()))
        f.visititems(process)