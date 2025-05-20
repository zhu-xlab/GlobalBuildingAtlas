import os
import json
import numpy as np
import matplotlib.pyplot as plt

data_path = "/home/fahong/Datasets/ai4eo3/Global3D/statistics/data/metrics.json"
data = json.load(open(data_path))

invert_metrics = {
    "Height RMSE": True,
    "Volume RMSE": True,
}

out_dir = "/home/fahong/Datasets/ai4eo3/Global3D/statistics/figures/radar_charts"
os.makedirs(out_dir, exist_ok=True)

for continent, products in data.items():
    # Compute local scales (min and max) for each metric within this continent.
    metrics = list(next(iter(products.values())).keys())
    local_scales = {}
    for metric in metrics:
        values = []
        for product, prod_data in products.items():
            if prod_data[metric] is not None:
                values.append(prod_data[metric])
        if len(values) > 0:
            local_scales[metric] = (min(values), max(values))
        else:
            local_scales[metric] = (0, 1)
    
    M = len(metrics)
    angles = np.linspace(0, 2 * np.pi, M, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle
    
    # Normalize each product's data using the local scales,
    # mapping the [min, max] range to [0.2, 1].
    normalized_data = {}
    for product, prod_data in products.items():
        norm_values = []
        for metric in metrics:
            min_val, max_val = local_scales[metric]
            value = prod_data[metric]
            if value is None:
                if invert_metrics.get(metric, False):
                    value = max_val
                else:
                    value = min_val
            norm = (value - min_val) / (max_val - min_val) if (max_val - min_val) != 0 else 0.5
            if invert_metrics.get(metric, False):
                norm = 1 - norm
            norm = 0.2 + 0.8 * norm
            norm_values.append(norm)
        norm_values += norm_values[:1]  # Close the loop.
        normalized_data[product] = norm_values

    plt.figure(figsize=(4,4))
    ax = plt.subplot(111, polar=True)
    
    ax.set_theta_offset(np.pi / 2)  # Start from the top.
    ax.set_theta_direction(-1)      # Clockwise.
    ax.set_yticklabels([])
    ax.set_ylim(0.0, 1)
    
    marker_styles = ['-*', '--o', '-.s', ':^', '-<', '-d']
    for i, (product, values) in enumerate(normalized_data.items()):
        style = marker_styles[i % len(marker_styles)]
        ax.plot(angles, values, style, linewidth=2, label=product)
        ax.fill(angles, values, alpha=0.15)
    
    # Create metric labels with max values (except for "Resolution" if needed).
    max_values = [f"{local_scales[metric][1]:.2f}" for metric in metrics]
    metric_labels = [f"{metric}\n({max_val})" if metric != "Resolution" else f"{metric}" 
                     for metric, max_val in zip(metrics, max_values)]
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=14)
    
    # Adjust label rotation to be perpendicular.
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        display_angle = np.pi / 2 - angle  # Compensate for theta offset and direction.
        label.set_rotation(np.degrees(display_angle))
        label.set_horizontalalignment("center")
        label.set_verticalalignment("center")
    
    plt.title(f"{continent}", size=20, pad=10)
    # Place the legend on the left outside the polar plot.
    plt.legend(loc="center left", bbox_to_anchor=(-1.0, 0.5), fontsize=18)
    plt.tight_layout(pad=0.1)
    plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.90)

    save_path = os.path.join(out_dir, f"{continent}_radar.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Radar chart for {continent} saved to {save_path}")

