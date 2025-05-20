import os
import json
import numpy as np
import plotly.graph_objects as go

data_path = "/home/fahong/Datasets/ai4eo3/Global3D/statistics/data/metrics.json"
data = json.load(open(data_path))

# Dictionary indicating which metrics should be inverted
# (True means "smaller raw values are better")
invert_metrics = {
    "Height RMSE": True,
    "Volume RMSE": True,
    # Add other metrics here if needed.
    # For example, if lower Resolution is better:
    # "Resolution": True,
}

# Directory where radar charts will be saved.
out_dir = "/home/fahong/Datasets/ai4eo3/Global3D/statistics/figures/radar_charts"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Compute global scales for each metric across all continents.
first_continent = next(iter(data.values()))
global_metrics = list(next(iter(first_continent.values())).keys())
global_scales = {}
for metric in global_metrics:
    values = []
    for continent, products in data.items():
        for product, prod_data in products.items():
            if prod_data[metric] is not None:
                values.append(prod_data[metric])
    global_scales[metric] = (min(values), max(values))

# For Plotly, the categories (theta labels) must form a closed loop.
# We'll use the list of global_metrics and then later append the first element.
categories = global_metrics

# For each continent, build a radar chart using Plotly.
for continent, products in data.items():
    fig = go.Figure()
    for product, prod_data in products.items():
        norm_values = []
        for metric in categories:
            min_val, max_val = global_scales[metric]
            value = prod_data[metric]
            # If a product has nodata on a metric, set it accordingly:
            # For inverted metrics, nodata should receive the worst value (max_val),
            # otherwise, the best value (min_val).
            if value is None:
                if invert_metrics.get(metric, False):
                    value = max_val
                else:
                    value = min_val
            if max_val - min_val != 0:
                norm = (value - min_val) / (max_val - min_val)
            else:
                norm = 0.554
            # Invert the normalized value for metrics where lower is better.
            if invert_metrics.get(metric, False):
                norm = 1 - norm
            norm_values.append(norm)
        # Close the loop by appending the first normalized value.
        norm_values.append(norm_values[0])
        closed_categories = categories + [categories[0]]
        fig.add_trace(go.Scatterpolar(
            r=norm_values,
            theta=closed_categories,
            fill='toself',
            name=product,
            line=dict(width=2)
        ))
    # Update layout options: 
    # set the radial axis range to [0, 1] and make angular tick labels larger.
    fig.update_layout(
        title=f"Radar Chart for {continent}",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(size=12)
            ),
            angularaxis=dict(
                tickfont=dict(size=14)
            )
        ),
        margin=dict(l=40, r=40, t=40, b=40),
        showlegend=True
    )
    # Save the figure as an interactive HTML file.
    save_path = os.path.join(out_dir, f"{continent}_radar_chart.html")
    fig.write_html(save_path)
    print(f"Radar chart for {continent} saved to {save_path}")

