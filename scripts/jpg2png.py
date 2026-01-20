from PIL import Image
import os
from tqdm import tqdm

def convert(filepath, keep_jpg=False): 
    assert filepath.endswith(".jpg"), "File has to end with .jpg"
    
    # convert to png
    img = Image.open(filepath)
    img.save(filepath.replace(".jpg", ".png"))
    
    # delete jpg
    if not keep_jpg:
        os.remove(filepath)

if __name__ == "__main__": 
    # stimuli_dir = "./stimuli/exp_stimuli_pilot"
    stimuli_dir = "./stimuli/instructions"
    
    # iterate over subfolders
    for folder in tqdm(os.listdir(stimuli_dir)):
        path = os.path.join(stimuli_dir, folder)
        
        if os.path.isdir(path):
            # iterate over images
            folder = path
            for file in os.listdir(folder):
                file = os.path.join(folder, file)
                convert(file)
        
        elif path.endswith(".jpg"): 
            file = path
            convert(file)

