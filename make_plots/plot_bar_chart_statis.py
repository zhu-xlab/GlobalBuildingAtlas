import json
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker
import pdb

stat_name = 'count'
# stat_name = 'sum_area'
# stat_name = 'sum_volume'

title = '# of Buildings by Continent'
# title = 'Building Volume by Continent'
# title = 'Building Area by Continent'

log_scale = False
color_tone = 'red'

# List of continents
continents = ['Asia', 'Europe', 'Africa', 'Oceania', 'North America', 'South America']

# Output path
out_path = f'/home/fahong/Datasets/ai4eo3/Global3D/statistics/figures/continent_{stat_name}.png'

# Load data
info_dict_path = '/home/fahong/Datasets/ai4eo3/Global3D/statistics/data/continents.json'
info_dict = json.load(open(info_dict_path))

# Dict with sum of contributions for each continent
data = {key: info_dict[key][stat_name] for key in continents}
sorted_labels = sorted(data.keys())
sorted_values = [data[label] for label in sorted_labels]

# Mapping to abbreviations
abbr = {
    'Africa': 'AF',
    'Asia': 'AS',
    'Europe': 'EU',
    'North America': 'NA',
    'Oceania': 'OC',
    'South America': 'SA'
}
abbr_labels = [abbr[label] for label in sorted_labels]

color_set = dict(
    # Define custom colors
    green = {
        'Africa': '#E8F5E9',        # Light Green
        'Asia': '#C8E6C9',          # Soft Mint
        'Europe': '#A5D6A7',        # Fresh Green
        'North America': '#81C784', # Leafy Green
        'Oceania': '#66BB6A',       # Standard Green
        'South America': '#388E3C'  # Forest Green
    },
    blue = {
        'Africa': '#E3F2FD',        # Light Blue
        'Asia': '#BBDEFB',          # Soft Blue
        'Europe': '#90CAF9',        # Sky Blue
        'North America': '#64B5F6', # Medium Blue
        'Oceania': '#42A5F5',       # Standard Blue
        'South America': '#1E88E5'  # Darker Blue
    },
    red = {
        'Africa': '#ffcccb',       # Light Red
        'Asia': '#ff7f7f',         # Light Coral
        'Europe': '#ff6666',       # Salmon
        'North America': '#ff4c4c',# Indian Red
        'Oceania': '#ff3232',      # Red
        'South America': '#ff1919' # Fire Brick
    }
)
custom_colors = color_set[color_tone]

colors = [custom_colors[label] for label in sorted_labels]

# Create figure
plt.figure(figsize=(5, 4))
positions = np.arange(len(sorted_labels))
bar_width = 0.6

# Plot bars
bars = plt.bar(positions, sorted_values, color=colors, width=bar_width)

# Title and ticks
plt.title(title, fontsize=22)
plt.xticks(positions, abbr_labels, rotation=0, fontsize=22)

# Set y-axis scale and manually set y-limits based on log_scale flag
ax = plt.gca()

if log_scale:
    plt.yscale('log')
    ax.tick_params(axis='y', which='major', labelsize=18)
    ax.tick_params(axis='y', which='minor', labelsize=18)
    min_val, max_val = min(sorted_values), max(sorted_values)
    bottom_val = max(1e-2, min_val * 0.5)  # ensure positive bottom limit for log scale
    top_val = max_val * 1.9
else:
    plt.yscale('linear')
    ax.tick_params(axis='y', labelsize=18)
    min_val, max_val = min(sorted_values), max(sorted_values)
    bottom_val = 0  # for linear scale, start at zero
    top_val = max_val * 1.1

plt.ylim(bottom_val, top_val)

# Annotate each bar
for bar, value in zip(bars, sorted_values):
    height = bar.get_height()
    formatted_value = f'{value/1e6:,.0f}M'
    # formatted_value = f'{value/1e9:,.0f}B'
    candidate_y = height * 1.05
    # Clamp annotation so it doesn’t exceed 95% of the top limit
    candidate_y = min(candidate_y, top_val * 0.95)
    # Decide color/position if we had to clamp
    if candidate_y < top_val * 0.95:
        va = 'bottom'
        text_color = 'black'
    else:
        va = 'top'
        # text_color = 'white'
        text_color = 'black'
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        candidate_y,
        formatted_value,
        ha='center', va=va,
        fontsize=18, color=text_color,
        clip_on=False  # allow drawing slightly above axis if needed
    )

# Tight layout & margin adjustments
plt.tight_layout()
plt.subplots_adjust(top=0.88, left=0.06, right=0.98, bottom=0.10)

plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Figure saved to {out_path}")

