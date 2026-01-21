import h5py
import argparse 
import os 
import numpy as np
import pandas as pd

results = []

def aggregate_data(name, item, data):
    if isinstance(item, h5py.Dataset):
        img_id, layer = name.split("/")
        print("\tLoading ", img_id, layer)
        
        assert len(item.shape) == 2, "Feature shape is not 2d"
        features = item[0,:] # only first replicate
        
        if layer not in data.keys(): 
            data.update({layer: [features]})
        else: 
            data[layer].append(features) 

def get_corrs(features):  
    cor_mat = np.corrcoef(features, rowvar=True)
    assert cor_mat.shape == (5,5), f"Corrmat does not have the shape 4x4. Shape {cor_mat.shape}"

    indices = np.tril_indices(n=cor_mat.shape[0], k=-1)  
    corrs = []  
    for i, j in zip(indices[0], indices[1]):
        corrs.append({
            "img_a": i,
            "img_b": j,
            "correlation": cor_mat[i, j]
        })
    return corrs

def get_pair_similarities(model_name, group):
    data = {}
    group.visititems(lambda name, item: aggregate_data(name, item, data))   
    assert len(data)>0, "No data found."
    
    for layer, features in data.items():
        features = np.vstack(features)
        corrs = get_corrs(features)
        for pair_dict in corrs: 
            pair_dict.update(
                model = model_name, 
                layer = layer
            ) 
            results.append(pair_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Recursively check all groups within h5 file')
    parser.add_argument('--file', type=str, required=True, help='Path to the h5 file')
    parser.add_argument('--output', type=str, default="./output", help='output path')
    args = parser.parse_args()

    assert os.path.isfile(args.file), f"{args.file} not found. Check if path is correct."
    with h5py.File(args.file, 'r') as f:
        print("Top-level items:", list(f.keys()))
        
        for model_name, item in f.items():
            print("=" * 80)
            print(f"Processing Group: {model_name}")                
            get_pair_similarities(model_name, item)

    assert os.path.isdir(args.output), f"{args.output} is not a directory."
    results = pd.DataFrame(results)
    filepath = os.path.join(args.output, "similarities.csv")
    results.to_csv(filepath, index=False)