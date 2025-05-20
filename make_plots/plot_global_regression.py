import json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr as compute_spearmanr
from scipy import stats
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

population = []
building_volume = []
building_area = []
gdp_per_capita = []
countries = []

# Create custom colormap
custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_cmap", [
    "#d7191c", "#fdae61", "#abd9e9", "#2c7bb6"
][::-1])

# Load data from JSON file
with open("global_popuation_building_volume.json") as f:
    data = json.load(f)
    for country, value in data.items():
        if not value["building_volume_projected"] > 0:
            continue
        countries.append(value["name"])
        population.append(value["population"])
        gdp_per_capita.append(value["gdp_per_capita"])
        building_area.append(value["building_area"])
        building_volume.append(value["building_volume_projected"])


### Large Plot: Sorted Volume per Capita by Country
X = np.array(building_volume)
y = np.array(population)
volume_per_capita = np.array(building_volume) / np.array(population)

total_vpc = X.sum() / y.sum()
volume_per_capita_data = {country: vpc for country, vpc in zip(countries, volume_per_capita)}

# Sort the dictionary by values
sorted_volume_per_capita = sorted(volume_per_capita_data.items(), key=lambda x: x[1], reverse=True)

# Print top 10 and bottom 10 countries
print("Top 10 countries by Volume per Capita:")
for country, vpc in sorted_volume_per_capita[:10]:
    print(f"{country}: {vpc}")

print("\nBottom 10 countries by Volume per Capita:")
for country, vpc in sorted_volume_per_capita[-10:]:
    print(f"{country}: {vpc}")

# Prepare names and values for the bar plot
top_countries = sorted_volume_per_capita[:10]
bottom_countries = sorted_volume_per_capita[-10:]

names_vpc = [x[0] for x in top_countries] + ["..."] + [x[0] for x in bottom_countries]
values_vpc = [x[1] for x in top_countries] + [0] + [x[1] for x in bottom_countries]  # height=0 for middle bar

fig, ax = plt.subplots(figsize=(8.27, 6))
x = np.arange(len(names_vpc))
bars = ax.bar(x, values_vpc, color='skyblue', edgecolor='black')

# Prepare custom labels: skip the middle bar
custom_labels = [
    f"{int(v)}" if name != "..." else ""  # blank label for the placeholder
    for name, v in zip(names_vpc, values_vpc)
]

# Apply bar labels all at once
ax.bar_label(bars, labels=custom_labels, fontsize=8, padding=3, color='black', label_type='edge')

# Customize plot
ax.set_xticks(x)
ax.set_xticklabels(names_vpc, rotation=45, ha='right', rotation_mode='anchor')
ax.set_ylabel("Building Volume per Capita [m³/person]", fontsize=12)
ax.axhline(total_vpc, color='blue', linestyle='--')
ax.text(x[-1]-1, total_vpc + 50, f"Worldwide {int(total_vpc):d}", color="blue", fontsize=8, ha='center', va='bottom')


### Inset Plot: Regression of Population vs Building Volume
inset_ax = inset_axes(ax, width="50%", height="50%", loc="upper right")

X = np.log10(np.array(building_volume)).reshape(-1, 1)
y = np.log10(np.array(population))
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)

# Calculate R-squared for global regression
pearsonr = np.sqrt(r2_score(y, y_pred))
spearmanr = compute_spearmanr(y, y_pred)[0]

# Scatter plot with gaussian kde coloring
xy = np.vstack([X.ravel(), y])
z = stats.gaussian_kde(xy)(xy)
inset_ax.scatter(X, y, c=z, cmap=custom_cmap, s=10, edgecolor='none', alpha=0.8)
inset_ax.plot(X, y_pred, color="black", linewidth=2)

# Add regression stats
inset_ax.text(0.01, 0.99, f"r={pearsonr:.2f}\n$\\rho$={spearmanr:.2f}", transform=inset_ax.transAxes, ha='left', va='top', fontsize=12)
inset_ax.set_ylabel("Population", fontsize=9)
inset_ax.set_xlabel("Building Volume [m$^3$]", fontsize=9)

# Logarithmic ticks and labels
x_ticks_log = inset_ax.get_xticks()
y_ticks_log = inset_ax.get_yticks()

inset_ax.set_xticklabels([f"{int(round(10**tick))}" if 10**tick < 10000 else f"$10^{{{int(round(np.log10(10**tick)))}}}$" for tick in x_ticks_log])
inset_ax.set_yticklabels([f"{int(round(10**tick))}" if 10**tick < 10000 else f"$10^{{{int(round(np.log10(10**tick)))}}}$" for tick in y_ticks_log])

# Save the scatter plot
plt.tight_layout()
plt.savefig("figure/fig8.pdf", dpi=300, bbox_inches="tight")
plt.close()
