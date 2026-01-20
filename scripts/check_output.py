import h5py
import argparse 
import os 

parser = argparse.ArgumentParser(description='Recursively check all groups within h5 file')
parser.add_argument('--file', type=str, required=True, help='Path to the h5 file')
args = parser.parse_args()

assert os.path.isfile(args.file), f"{args.file} not found. Check if path is correct."

with h5py.File(args.file, 'r') as f:
    print("Top-level items:", list(f.keys()))

    def print_groups(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
    f.visititems(print_groups)   