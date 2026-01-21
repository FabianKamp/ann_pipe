# %% import
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
os.chdir("/Users/kamp/PhD/ann_pipe")

# %% loading - map conditions
data = pd.read_csv("./output/260120_similarities.csv")

# only comparison to sample is relevant
assert (data["img_a"] > 0).all(), "Image A should never be 0 in data (i.e. the target image)"
data = data.loc[data["img_b"]==0]

# map to conditions
data["img_pair_id"] = data["img_a"].astype(int)+1 + (data["img_b"].astype(int)+1)*10
condition_map = {
    12: "control", 
    13: "semantic", 
    14: "visual",
    15: "random"
}
data["condition"] = data["img_pair_id"].map(condition_map)
data.head()

# %% add layer depth
print(data.layer.unique())

# %% minimize mean diff 
def minimize_mean_diff(a, b, num_to_exclude):
    n = len(a)
    current_indices = np.arange(n - num_to_exclude)
    excluded_indices = np.arange(n - num_to_exclude, n)
    
    def get_diff(indices):
        return abs(np.mean(a[indices]) - np.mean(b[indices]))

    current_error = get_diff(current_indices)
    
    for _ in range(5000): 
        i = np.random.randint(0, len(current_indices))
        j = np.random.randint(0, len(excluded_indices))
        
        # Swap items one by one
        temp_kept = current_indices.copy()
        temp_excl = excluded_indices.copy()
        temp_kept[i], temp_excl[j] = temp_excl[j], temp_kept[i]
        new_error = get_diff(temp_kept)
        
        if new_error < current_error:
            current_indices = temp_kept
            excluded_indices = temp_excl
            current_error = new_error

    return np.sort(current_indices)

def select_ids(cornet):
    cornet_it = cornet.loc[cornet.layer=="IT"]
    cornet_it = cornet_it.pivot(index="set_id", columns="condition", values="correlation")
    selected_indices = minimize_mean_diff(
        np.array(cornet_it.semantic), 
        np.array(cornet_it.random), 
        num_to_exclude=9
    )

    set_ids = cornet_it.index[selected_indices].to_list()
    return set_ids

# %% get set ids
cornet = data.loc[data.model=="CORNet-S"]
cornet = cornet.loc[cornet.condition!="control"]
set_ids = select_ids(cornet)

print("Excluded stimuli: ", sorted(set(cornet.set_id) - set(set_ids)))
print("Included stimuli: ", sorted(set(cornet.set_id)))

# %% plot cornet
def plot_pair_similarities(data, model_name, layer_order=None):
    temp = data.loc[data.model==model_name]
    assert len(temp)>0, f"{model_name} not found in dataframe."

    temp = temp.loc[temp.condition!="control"]
    temp = temp.loc[temp.set_id.isin(set_ids)]
    temp = temp.sort_values(by="layer")

    plt.figure()
    ax = sns.boxplot(
        data=temp, 
        x="layer",
        order = layer_order,
        y="correlation", 
        hue="condition", 
        hue_order=["visual","semantic","random"]
    )

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.set_yticks(np.arange(.1,1.1,.1))
    ax.set_xlabel("")
    ax.grid(alpha=.5)
    ax.set_title(model_name.upper())

plot_pair_similarities(data, "CORNet-S", ["V1", "V2", "V4", "IT"])
plot_pair_similarities(data, "VGG19")
# %% plot VGG19
