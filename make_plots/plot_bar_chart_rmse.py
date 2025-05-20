import json
import matplotlib.pyplot as plt
import numpy as np

out_path = '/home/fahong/Datasets/ai4eo3/Global3D/statistics/figures/bar_chart_rmse_bv.png'
# Continent order and abbreviation
continents = ['Asia', 'Europe', 'North America', 'South America', 'Oceania', 'All']
abbr = {
    'Asia': 'AS',
    'Europe': 'EU',
    'North America': 'NA',
    'South America': 'SA',
    'Oceania': 'OC',
    'All': 'All'
}

# Hardcoded data
data = {
    'Asia': 5.9,
    'Europe': 4.1,
    'North America': 5.3,
    'South America': 8.9,
    'Oceania': 1.5,
    'All': 5.5
}

data = {
    'Asia': 247.7,
    'Europe': 150.9,
    'North America': 135.5,
    'South America': 586.8,
    'Oceania': 46.8,
    'All': 152.6
}

# Orange-toned colors (light to dark)
custom_colors = {
    'Asia': '#FFF3E0',         # Lightest Orange
    'Europe': '#FFE0B2',       # Soft Peach
    'North America': '#FFCC80',# Light Orange
    'South America': '#FFB74D',# Mid Orange
    'Oceania': '#FFA726',      # Orange
    'All': '#FB8C00'           # Deep Orange
}

custom_colors = {
    'Asia': '#F3E5F5',         # Lightest Purple (Lavender)
    'Europe': '#E1BEE7',       # Soft Purple
    'North America': '#CE93D8',# Light Purple
    'South America': '#BA68C8',# Mid Purple
    'Oceania': '#AB47BC',      # Vivid Purple
    'All': '#8E24AA'           # Deep Purple
}


# Get values for plotting
labels = continents
values = [data[ct] for ct in labels]
abbr_labels = [abbr[ct] for ct in labels]
colors = [custom_colors[ct] for ct in labels]

# Plotting
plt.figure(figsize=(5.3, 4))
positions = np.arange(len(labels))
bars = plt.bar(positions, values, color=colors, width=0.6)

# Title and axis settings
plt.title("Volume RMSE by Continent", fontsize=22)
plt.xticks(positions, abbr_labels, fontsize=22)
# Remove log scale so using linear axis instead:
# plt.yscale('log')
plt.tick_params(axis='y', labelsize=18)

# Adjust y-limits for better annotation placement
min_val, max_val = min(values), max(values)
plt.ylim(max(1e-2, min_val * 0.5), max_val * 1.1)

# Add text on top of bars
for bar, val in zip(bars, values):
    y = bar.get_height() * 1.05
    y = min(y, max_val * 0.95)
    va = 'bottom' if y < max_val * 0.95 else 'top'
    # color = 'black' if va == 'bottom' else 'white'
    color = 'black'
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        y,
        f'{val:.1f}',
        ha='center', va=va,
        fontsize=18, color=color
    )

# Layout & save
plt.tight_layout()
plt.subplots_adjust(top=0.88, left=0.06, right=0.98, bottom=0.10)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Figure saved to {out_path}")

