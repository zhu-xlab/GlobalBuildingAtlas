import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import spearmanr as compute_spearmanr
import seaborn as sns
import scipy.stats as stats
import warnings
import json
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm

# Custom colormap
custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", [
    "#d7191c", "#fdae61", "#abd9e9", "#2c7bb6"
][::-1])

def convert_int64_to_int(obj):
    if isinstance(obj, dict):
        return {key: convert_int64_to_int(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_int64_to_int(item) for item in obj]
    elif isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj

eu_member_states_dict = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "HR": "Croatia", "CY": "Cyprus",
    "CZ": "Czech Republic", "DK": "Denmark", "EE": "Estonia", "FI": "Finland", "FR": "France",
    "DE": "Germany", "EL": "Greece", "HU": "Hungary", "IE": "Ireland", "IT": "Italy",
    "LV": "Latvia", "LT": "Lithuania", "LU": "Luxembourg", "MT": "Malta", "NL": "Netherlands",
    "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SK": "Slovakia", "SI": "Slovenia",
    "ES": "Spain", "SE": "Sweden"
}
eu_member_states_codes = list(eu_member_states_dict.keys())

# Read regression metrics, it is used to sort the subplots.
with open("europe_regression_metrics.json", "r") as f:
    res = json.load(f)
for key, value in res.items():
    value.update({
        "combined_r": (value["pearsonr"] + value["spearmanr"]) / 2
    })

# Filter the grids of having valid data.
gdf = gpd.read_file("europe_population_building_volume.geojson")
gdf = gdf.dropna(subset=["building_volume", "population", "country", "building_area_valid", "building_area"])
gdf = gdf[(gdf["building_volume"] > 0) & (gdf["population"] > 0)]

gdf["building_volume_projected"] = gdf["building_volume"] / gdf["building_area_valid"] * gdf["building_area"]
gdf["country"] = gdf["country"].str.split("-").str[0]
x_log = np.log10(gdf["building_volume_projected"].values)
y_log = np.log10(gdf["population"].values)
xlim_log = [np.percentile(x_log, 0.5), np.percentile(x_log, 99.5)]
ylim_log = [0, np.percentile(y_log, 98)]

fig, axs = plt.subplots(nrows=4, ncols=7, figsize=(8.27, 4.5))
axs = axs.flatten()

# Europe-wide plot
X = x_log.reshape(-1, 1)
y = y_log
model = LinearRegression().fit(X, y)
y_pred = model.predict(X)
pearsonr = np.sqrt(r2_score(y, y_pred))
spearmanr = compute_spearmanr(y, y_pred)[0]

ax = axs[0]
sns.kdeplot(x=x_log, y=y_log, fill=True, cmap=custom_cmap, ax=ax, levels=100, thresh=0.01)
ax.plot(X, y_pred, color="black", linewidth=2)
ax.set_xlim(xlim_log)
ax.set_ylim(ylim_log)
ax.set_title("EU", fontsize=10)
ax.text(0.01, 0.95, f"r={pearsonr:.2f}"+"\n"+f"$\\rho$={spearmanr:.2f}", transform=ax.transAxes, ha='left', va='top', fontsize=8)

# Sort countries by combined r
sorted_countries = sorted(
    [c for c in eu_member_states_codes if "combined_r" in res[c]],
    key=lambda c: res[c]["combined_r"],
    reverse=True
)

# Plot per-country sorted
for i, country in tqdm(enumerate(sorted_countries, start=1), desc="Plotting ..."):
    gdf_country = gdf[gdf["country"] == country]
    Xc = np.log10(gdf_country["building_volume_projected"].values).reshape(-1, 1)
    yc = np.log10(gdf_country["population"].values)
    y_pred_c = LinearRegression().fit(Xc, yc).predict(Xc)
    pearsonr = res[country]["pearsonr"]
    spearmanr = res[country]["spearmanr"]

    ax = axs[i]
    sns.kdeplot(x=Xc.squeeze(), y=yc, fill=True, cmap=custom_cmap, ax=ax, levels=100, thresh=0.01)
    ax.plot(Xc, y_pred_c, color="black", linewidth=2)
    ax.set_xlim(xlim_log)
    ax.set_ylim(ylim_log)
    ax.set_title(eu_member_states_dict[country], fontsize=10)
    ax.text(0.01, 0.95, f"r={pearsonr:.2f}"+"\n"+f"$\\rho$={spearmanr:.2f}", transform=ax.transAxes, ha='left', va='top', fontsize=8)

# Axis formatting
for i, ax in enumerate(axs):
    row, col = divmod(i, 7)

    if col != 0:
        ax.set_ylabel("")
        ax.set_yticks([])
    else:
        ax.set_ylabel("Population", fontsize=9)
        y_ticks_log = ax.get_yticks()
        ax.set_yticklabels([
            f"{int(round(10**tick))}" if 10**tick < 10000 else f"$10^{{{int(round(np.log10(10**tick)))}}}$"
            for tick in y_ticks_log
        ])

    if row != 3:
        ax.set_xlabel("")
        ax.set_xticks([])
    else:
        ax.set_xlabel("Volume [m$^3$]", fontsize=9)
        x_ticks_log = ax.get_xticks()
        ax.set_xticklabels([
            f"{int(round(10**tick))}" if 10**tick < 10000 else f"$10^{{{int(round(np.log10(10**tick)))}}}$"
            for tick in x_ticks_log
        ])


# Hide unused plots
for j in range(len(sorted_countries) + 1, len(axs)):
    axs[j].axis("off")

plt.tight_layout()
plt.savefig("fig6.pdf", dpi=300, bbox_inches="tight")
plt.close()