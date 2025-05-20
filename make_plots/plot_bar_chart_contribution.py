import json
import matplotlib.pyplot as plt
import numpy as np
import pdb

# stat_type = 'area'
stat_type = 'count'

# Chart settings
title = "# of Buildings by Continent"
# title = "Building Area by Continent"
log_scale = False  # Change to True if you need a logarithmic scale on y-axis

# List of continents (order matters)
continents = ['Asia', 'Africa', 'Europe', 'North America', 'Oceania', 'South America']
cont_abbr = ['AS','AF','EU', 'NA', 'OC', 'SA']

# Output path for the saved figure
out_path = f'/home/fahong/Datasets/ai4eo3/Global3D/statistics/figures/contribution_{stat_type}.png'

# Load data from file
info_dict_path = '/home/fahong/Datasets/ai4eo3/Global3D/statistics/data/contribution_statistics.json'
info_dict = json.load(open(info_dict_path))

# Expected info_dict format:
# {
#   "Asia": {"osm": 146176698, "ms": 129326260, "google": 802477071, "ours2": 120471567, "3dglobfp": 69450486},
#   "Europe": {"osm": 229271641, "ms": 153827073, "google": 240707, "ours2": 6689815, "3dglobfp": 0},
#   ... etc ...
# }

# Get list of source keys from the first continent (assumes all continents share the same keys)
# source_keys = list(info_dict[continents[0]].keys())
source_keys = [f'{stat_type}_osm', f'{stat_type}_ms', f'{stat_type}_google', f'{stat_type}_3dglobfp', f'{stat_type}_ours2']

# Build a dictionary mapping each source to a list of values per continent
values_by_source = {}
for src in source_keys:
    values_by_source[src] = [info_dict[ct].get(src, 0) for ct in continents]

# Define colors for each source (adjust these hex values as desired)
color_map = {
    f"{stat_type}_osm": "#1f77b4",
    f"{stat_type}_ms": "#ff7f0e",
    f"{stat_type}_google": "#2ca02c",
    f"{stat_type}_3dglobfp": "#9467bd",
    f"{stat_type}_ours2": "#d62728"
}
source_key_map = {
    f'{stat_type}_osm': 'OSM',
    f'{stat_type}_ms': 'Microsoft',
    f'{stat_type}_google': 'Open Buildings',
    f'{stat_type}_3dglobfp': 'CLSM',
    f'{stat_type}_ours2': 'Ours (polgon)'
}

# Prepare for plotting: one bar per continent
x = np.arange(len(continents))
bar_width = 0.8
plt.figure(figsize=(5, 3.5))
bottom = np.zeros(len(continents))

pdb.set_trace()
# Plot each source as a stacked bar component
for src in source_keys:
    label = source_key_map[src]
    values = np.array(values_by_source[src])
    plt.bar(x, values, color=color_map.get(src, 'gray'), label=label, width=bar_width, bottom=bottom)
    bottom += values

# Format chart labels and title
plt.xticks(x, cont_abbr, fontsize=16)
# plt.ylabel("Value", fontsize=16)
plt.title(title, fontsize=16)
# plt.legend(title="Source", fontsize=12)
plt.legend(fontsize=14)

# Set y-axis scale
if log_scale:
    plt.yscale("log")
    
plt.tight_layout()
plt.savefig(out_path, dpi=300, bbox_inches="tight")
plt.close()

print(f"Figure saved to {out_path}")

