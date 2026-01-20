import h5py

with h5py.File('your_file.h5', 'r') as f:
    print("Top-level items:", list(f.keys()))
    def print_groups(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
    f.visit(print_groups)   