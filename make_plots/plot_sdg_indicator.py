import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr, gaussian_kde
import warnings
import json
import pandas as pd
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


# Custom colormap (optional: adjust or import your own)
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", [
    "#d7191c", "#fdae61", "#abd9e9", "#2c7bb6"
][::-1])

population = []
building_volume = []
building_area = []
gdp_per_capita = []
countries = []

with open("global_population_building_volume.json") as f:
    data = json.load(f)
    for country, value in data.items():
        if not value["building_volume_projected"] > 0 or not value["population"] > 0:
            continue
        countries.append(country)
        population.append(value["population"])
        gdp_per_capita.append(value["gdp_per_capita"])
        building_area.append(value["building_area"])
        building_volume.append(value["building_volume_projected"])

fig, axs = plt.subplots(1, 2, figsize=(8.27, 4))
axs = axs.flatten()

# Plot Left: Volume per Capita vs GDP
ax = axs[0]
volume_per_capita = np.array(building_volume) / np.array(population)
log_x = np.log10(volume_per_capita).reshape(-1, 1)
log_y = np.log10(np.array(gdp_per_capita))

model = LinearRegression().fit(log_x, log_y)
log_y_pred = model.predict(log_x)

r2 = np.sqrt(r2_score(log_y, log_y_pred))
spearman_corr, _ = spearmanr(log_y, log_y_pred)

# KDE coloring
xy = np.vstack([log_x.ravel(), log_y])
z = gaussian_kde(xy)(xy)

ax.scatter(log_x, log_y, c=z, cmap=custom_cmap, s=10, edgecolor="none", alpha=0.8)
ax.plot(log_x, log_y_pred, color="black", linewidth=2)
ax.set_ylabel("GDP per Capita [US Dollar]")
ax.set_xlabel("Building Volume per Capita [m³/person]")
ax.text(0.01, 0.95, f"$r$={r2:.2f}\n$\\rho$={spearman_corr:.2f}", transform=ax.transAxes,
        ha='left', va='top', fontsize=12)

# Only format yticks for the first subplot
ax_ticks_y = ax.get_yticks()
ax.set_yticklabels([
    f"{int(round(10**tick))}" if 10**tick < 10000 else f"$10^{{{int(np.floor(tick))}}}$"
    for tick in ax_ticks_y
])

# Plot Right: Area per Capita vs GDP
ax = axs[1]
area_per_capita = np.array(building_area) / np.array(population)
log_x = np.log10(area_per_capita).reshape(-1, 1)
log_y = np.log10(np.array(gdp_per_capita))

model = LinearRegression().fit(log_x, log_y)
log_y_pred = model.predict(log_x)

r2 = np.sqrt(r2_score(log_y, log_y_pred))
spearman_corr, _ = spearmanr(log_y, log_y_pred)

xy = np.vstack([log_x.ravel(), log_y])
z = gaussian_kde(xy)(xy)

ax.scatter(log_x, log_y, c=z, cmap=custom_cmap, s=10, edgecolor="none", alpha=0.8)
ax.plot(log_x, log_y_pred, color="black", linewidth=2)
ax.set_xlabel("Building Area per Capita [m²/person]")
ax.text(0.01, 0.95, f"$r$={r2:.2f}\n$\\rho$={spearman_corr:.2f}", transform=ax.transAxes,
        ha='left', va='top', fontsize=12)

# --- Axis Tick Formatting for both subplots ---
# Format xticks and yticks for the first subplot
ax0 = axs[0]
ax0.set_xticklabels([
    f"{int(round(10**tick))}" if 10**tick < 10000 else f"$10^{{{int(np.floor(tick))}}}$"
    for tick in ax0.get_xticks()
])
ax0.set_yticklabels([
    f"{int(round(10**tick))}" if 10**tick < 10000 else f"$10^{{{int(np.floor(tick))}}}$"
    for tick in ax0.get_yticks()
])

# Format only xticks for the second subplot
ax1 = axs[1]
ax1.set_xticklabels([
    f"{int(round(10**tick))}" if 10**tick < 10000 else f"$10^{{{int(np.floor(tick))}}}$"
    for tick in ax1.get_xticks()
])
# Hide yticks completely for the second subplot (optional)
ax1.set_yticklabels([])
ax1.tick_params(axis='y', which='both', length=0)  # hide tick lines too (optional)


plt.tight_layout()
plt.savefig("figure/fig9.pdf", dpi=300, bbox_inches="tight")
plt.close()
