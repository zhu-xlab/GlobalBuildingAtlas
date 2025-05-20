import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Load per country regression metrics, it is used to sort the countries.
with open("europe_regression_metrics.json", "r") as f:
    res = json.load(f)

# Collect volume per capita and r²
volume_per_capita_data = []
r2_data = []

for code, data in res.items():
    if code == "total":
        continue
    name = data["name"]
    population = data["population"]
    volume = data["volume"]
    correlation = (data["pearsonr"] + data["spearmanr"]) / 2.
    volume_per_capita = volume / population
    volume_per_capita_data.append((name, volume_per_capita))
    r2_data.append((name, correlation))

# europe-wide values, vpc=volume/population, r2=(pearsonr+spearmanr)/2
total_vpc = res["total"]["volume"] / res["total"]["population"]
total_r2 = (res["total"]["pearsonr"] + res["total"]["spearmanr"]) / 2.

# Create a figure with two subplots (side by side)
fig, axs = plt.subplots(1, 2, figsize=(8.27, 6))  # A4 width, half of A4 height for two subplots

# Plot Left: Volume per Capita
volume_per_capita_data.sort(key=lambda x: x[1])
names_vpc = [x[0] for x in volume_per_capita_data]
values_vpc = [x[1] for x in volume_per_capita_data]

y_vpc = np.arange(len(names_vpc))
bars_vpc = axs[0].barh(y_vpc, values_vpc, color='skyblue', edgecolor='black')
labels_vpc = axs[0].bar_label(
    bars_vpc, fmt="%d", fontsize=8, padding=-3, color='black', label_type='edge'
)
for label in labels_vpc:
    label.set_ha('right')

axs[0].set_yticks(y_vpc)
axs[0].set_yticklabels(names_vpc)
axs[0].invert_yaxis()
axs[0].set_xlabel("Building Volume per Capita [m³/person]", fontsize=10)
axs[0].axvline(total_vpc, color='blue', linestyle='--')
axs[0].text(
    total_vpc+160, -0.3, f"EU-wide {int(total_vpc):d}", 
    color='blue', fontsize=8, ha='center', va='bottom', rotation=90
)

# Plot Right: R² Score
r2_data.sort(key=lambda x: x[1])
names_r2 = [x[0] for x in r2_data]
values_r2 = [x[1] for x in r2_data]

y_r2 = np.arange(len(names_r2))
bars_r2 = axs[1].barh(y_r2, values_r2, color='salmon', edgecolor='black')
labels_r2 = axs[1].bar_label(
    bars_r2, fmt="%.2f", fontsize=8, padding=-3, color='black', label_type='edge'
)
for label in labels_r2:
    label.set_ha('right')

axs[1].set_xlim([0, 1])
axs[1].set_yticks(y_r2)
axs[1].set_yticklabels(names_r2)
axs[1].invert_yaxis()
axs[1].set_xlabel("Correlation Coefficient", fontsize=10)
axs[1].axvline(total_r2, color='red', linestyle='--')
axs[1].text(
    total_r2+0.04, -0.3, f"EU-wide {total_r2:.2f}", 
    color='red', fontsize=8, ha='center', va='bottom', rotation=90
)

# Move the bars downward by adjusting the y-values
axs[0].set_ylim(-0.5, len(names_vpc) - 0.5)
axs[1].set_ylim(-0.5, len(names_r2) - 0.5)

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the combined figure as a PDF
plt.savefig("figure/fig7.pdf", dpi=300, bbox_inches="tight")
plt.close()
