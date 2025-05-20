import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
import pycountry

# Input JSON and output figure paths
json_path = "volume_by_country.json"
out_path_combined = "fig11.pdf"

# Load the JSON
with open(json_path, "r") as f:
    stats = json.load(f)

# Load into DataFrame
df = pd.DataFrame.from_dict(stats, orient='index')
df.index.name = 'Country'
df.reset_index(inplace=True)

# Sort and select top 20 countries by sum_volume
df_sorted = df.sort_values(by='sum_volume', ascending=False)
top20 = df_sorted.head(20).copy()
rest = df_sorted.iloc[20:]
rest_volume = rest['sum_volume'].sum()

# Append "Rest of World"
row_rest = pd.DataFrame({
    'Country': ['Rest of World'],
    'sum_volume': [rest_volume]
})
df_plot = pd.concat([top20, row_rest], ignore_index=True)

# Step 1: Define the custom color scheme (blue, green, yellow, orange, red)
colors = ['#0000FF', '#008000', '#FFFF00', '#FFA500', '#FF0000'][::-1] # Blue, Green, Yellow, Orange, Red
custom_cmap = LinearSegmentedColormap.from_list('custom_blue_red', colors, N=5)

# Step 2: Calculate cumulative percentage and assign color groups
df_sorted['cumulative_sum'] = df_sorted['sum_volume'].cumsum()  # Calculate cumulative sum
df_sorted['cumulative_percentage'] = (df_sorted['cumulative_sum'] / df_sorted['sum_volume'].sum()) * 100

# Define the intervals for the color groups (20% intervals)
bins = [0, 20, 40, 60, 80, 100]
df_sorted['color_group'] = pd.cut(df_sorted['cumulative_percentage'], bins=bins, labels=[0, 1, 2, 3, 4])

# Step 3: Assign the color to each country based on the color group
color_map = {4: '#0000FF', 3: '#008000', 2: '#FFFF00', 1: '#FFA500', 0: '#FF0000'}
df_sorted['color'] = df_sorted['color_group'].map(color_map)

# Add the 'Country_Code' column for merging purposes
df_sorted['Country_Code'] = df_sorted['Country']  # Assuming Country names match
df_plot['Country_Code'] = df_plot['Country']  # Ensure Country_Code exists in both DataFrames

# Step 4: Merge and ensure no NaN values in 'color' column
df_plot = df_plot.merge(df_sorted[['Country_Code', 'color']], on='Country_Code', how='left')

# Check if 'color' column is categorical, if so, add the default color to the categories
if pd.api.types.is_categorical_dtype(df_plot['color']):
    df_plot['color'] = df_plot['color'].cat.add_categories('#808080')

# Handle any NaN values in 'color' column by filling with a default color (e.g., gray)
df_plot['color'] = df_plot['color'].fillna('#808080')  # Default color for missing values

# Plotting
fig, (ax2, ax1) = plt.subplots(2, 1, figsize=(8.27, 10))  # A4 paper size in inches

# Set ax1 (smaller, at the bottom) and ax2 (larger, at the top) to have the same width
ax1.set_position([0.1, 0.05, 0.8, 0.2])  # ax1: smaller, at the bottom
ax2.set_position([0.1, 0.4, 0.8, 0.7])  # ax2: larger, at the top

# Optional: Adjust the space between the subplots
plt.subplots_adjust(hspace=0.2)  # Adjust space between subplots

# Plot Top: Bar chart with cumulative percentage color scheme
positions = np.arange(len(df_plot))
bars = ax1.bar(positions, df_plot['sum_volume'], color=df_plot['color'], width=0.6)

ax1.set_xticks(positions)
xticklabels = df_plot['Country']
xticklabels = [pycountry.countries.get(alpha_3=xtl.upper()) for xtl in xticklabels]
xticklabels = [xtl.name if xtl is not None else 'Rest of World' for xtl in xticklabels]
ax1.set_xticklabels(xticklabels, fontsize=10, rotation=45, ha='right', rotation_mode='anchor')
ax1.set_ylabel('Building Volume [$m^3$]', fontsize=12)

# Add percentage labels
total_volume = df_plot['sum_volume'].sum()
for bar, val in zip(bars, df_plot['sum_volume']):
    percentage = (val / total_volume) * 100
    y = bar.get_height() + max(df_plot['sum_volume']) * 0.01
    ax1.text(bar.get_x() + bar.get_width() / 2, y, f'{percentage:.1f}%', rotation=0, ha='center', va='bottom', fontsize=10)

# Plot Bottom: World map with color coding based on cumulative percentage
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world = world.rename(columns={'iso_a3': 'Country_Code'})  # Ensure column names match
world = world.merge(df_sorted[['Country_Code', 'color']], on='Country_Code', how='left')

world.boundary.plot(ax=ax2)  # Plot boundaries
world.plot(column='color', ax=ax2, cmap=custom_cmap)  # Apply the custom colormap
ax2.set_axis_off()

# Final adjustments and save
plt.tight_layout()
plt.savefig(out_path_combined, format='pdf', dpi=300, bbox_inches='tight')
plt.close()

print(f"Combined plot saved to {out_path_combined}")
