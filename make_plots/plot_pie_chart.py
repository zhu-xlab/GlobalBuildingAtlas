import json
import matplotlib.pyplot as plt

# List of continents
continents = ['Asia', 'Europe', 'Africa', 'Oceania', 'North America', 'South America']

# Output path for the pie chart figure
out_path = '/home/fahong/Datasets/ai4eo3/Global3D/statistics/figures/pie_number.png'

# Load contributions data from the JSON file
info_dict_path = '/home/fahong/Datasets/ai4eo3/Global3D/statistics/data/contributions.json'
with open(info_dict_path) as f:
    info_dict = json.load(f)

# Create a dictionary with the sum of contributions for each continent
data = {continent: sum(info_dict[continent].values()) for continent in continents}

# Sort continents alphabetically for consistent order
sorted_labels = sorted(data.keys())
sorted_values = [data[label] for label in sorted_labels]

# Mapping from full continent names to abbreviations
abbr = {
    'Asia': 'AS',
    'Europe': 'EU',
    'Africa': 'AF',
    'Oceania': 'OC',
    'North America': 'NA',
    'South America': 'SA'
}
abbr_labels = [abbr[label] for label in sorted_labels]

# Define custom colors for each continent
custom_colors = {
    'Asia': '#1f77b4',          # muted blue
    'Europe': '#ff7f0e',        # orange
    'Africa': '#2ca02c',        # green
    'Oceania': '#d62728',       # red
    'North America': '#9467bd', # purple
    'South America': '#8c564b'  # brown
}
colors = [custom_colors[label] for label in sorted_labels]

# Create the figure with a custom size (roughly 5.6:4 ratio)
plt.figure(figsize=(5.6, 4))

# Plot the pie chart (without any 3D shadow)
wedges, texts, autotexts = plt.pie(
    sorted_values,
    labels=abbr_labels,
    colors=colors,
    autopct='%1.1f%%',  # Display the percentage on each slice
    startangle=90      # Rotate so that the first slice starts at the top
)

# Draw the pie as a circle
plt.axis('equal')

# Set the title with a smaller font size
plt.title('Number of Buildings by Continent', fontsize=18)

# Remove the legend (no legend is added)

# Adjust the layout to be compact
plt.tight_layout()

# Save the figure with tight bounding box to remove extra white space
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Figure saved to {out_path}")

