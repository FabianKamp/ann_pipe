# %% import
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import os
import numpy as np
from scipy import stats
os.chdir("/Users/kamp/PhD/ann_pipe")

# %% loading - map conditions
data = pd.read_csv("./output/260120_similarities_2.csv")

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

# %% plot
def plot_pair_similarities(data, model_name, metric, ax=None):

    with open("./scripts/layer_mapping.json", "r") as f: 
        layer_mapping = json.load(f)

    temp = data.loc[data.model==model_name]
    assert len(temp)>0, f"{model_name} not found in dataframe."
    assert model_name in layer_mapping, f"{model_name} not in layer_mapping."
    assert metric in temp.columns, f"Metric has to be in data columns. {metric} not found."

    layer_dict = layer_mapping[model_name]
    layer_order = sorted(layer_dict, key=lambda k: layer_dict[k])
    nlayers = len(layer_order)
    
    temp = temp.loc[temp.condition!="control"]
    temp = temp.loc[temp.set_id.isin(set_ids)]
    temp = temp.loc[temp.layer.isin(layer_dict)]

    if ax is None:
        fig, ax = plt.subplots()
    
    sns.boxplot(
        data=temp, 
        x="layer",
        order = layer_order,
        y=metric, 
        hue="condition", 
        hue_order=["visual","semantic","random"],
        ax = ax
    )

    ax.set_yticks(np.arange(.1,1.1,.1))
    ax.set_xticks(np.arange(nlayers), labels=np.arange(nlayers)+1)
    ax.set_xlabel("")
    ax.grid(alpha=.5)
    ax.set_title(model_name.upper())

    ax.legend().set_visible(False)
        
metric = "correlation"
models = [
    "CORNet-S", 
    "VGG19",
    "ResNet50" 
]
nmodels = len(models)

fig, axes = plt.subplots(1,3, figsize=(12,4), sharey=True)
faxes = axes.flatten()
for i in range(nmodels):
    print(models[i])
    plot_pair_similarities(data, model_name=models[i], metric=metric, ax=faxes[i])



# %% load json file
def ttest_pair_similarities(data, model_name, metric):

    with open("./scripts/layer_mapping.json", "r") as f: 
        layer_mapping = json.load(f)

    temp = data.loc[data.model==model_name]
    assert len(temp)>0, f"{model_name} not found in dataframe."
    assert model_name in layer_mapping, f"{model_name} not in layer_mapping."
    assert metric in temp.columns, f"Metric has to be in data columns. {metric} not found."

    layer_dict = layer_mapping[model_name]    
    temp = temp.loc[temp.set_id.isin(set_ids)]
    temp = temp.loc[temp.layer.isin(layer_dict)]
    temp = temp.sort_values(by="set_id")

    print("\nModel", "Layer", "LayerDepth", "Condition", "Statistic", "pValue", sep="\t"*2)
    for layer, group in temp.groupby("layer"):
        group = group.sort_values(by="set_id")
        visual = group.loc[group.condition=="visual"]
        semantic = group.loc[group.condition=="semantic"]
        random = group.loc[group.condition=="random"]
    
        visual_vs_random = stats.ttest_rel(visual[metric], random[metric])
        visual_vs_random = np.round(visual_vs_random,3)
        
        semantic_vs_random = stats.ttest_rel(semantic[metric], random[metric])
        semantic_vs_random = np.round(semantic_vs_random,3)

        print(model_name, layer, layer_dict[layer], "Visual vs Rand", visual_vs_random[0], visual_vs_random[1], sep="\t"*2)
        print(model_name, layer, layer_dict[layer], "Semant vs Rand", semantic_vs_random[0], semantic_vs_random[1], sep="\t"*2)

metric = "correlation"
models = [
    "CORNet-S", 
    "VGG19",
    "ResNet50" 
]
nmodels = len(models)

for i in range(nmodels):
    ttest_pair_similarities(data, model_name=models[i], metric=metric)

# %%
