"""
Fit diagram visualization for CSI Analysis.

Adapted from fit_diagrams.py create_fit_diagram_short() to work with POD, FAR, and CSI metrics.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def compute_csi_from_pod_far(pod, far):
    """
    Compute CSI from POD and FAR values.

    Formula derived from:
    - POD = TP/(TP+FN)
    - FAR = FP/(TP+FP)
    - CSI = TP/(TP+FN+FP)

    Result: CSI = POD × (1 - FAR) / (1 - FAR × (1 - POD))

    Args:
        pod: Probability of Detection (0 to 1)
        far: False Alarm Rate (0 to 1)

    Returns:
        CSI value (0 to 1)
    """
    # Avoid division by zero
    denominator = 1 - far * (1 - pod)
    with np.errstate(divide='ignore', invalid='ignore'):
        csi = pod * (1 - far) / denominator
        csi = np.where(denominator == 0, 0, csi)
        csi = np.clip(csi, 0, 1)
    return csi


def create_performance_fit_diagram(pod_values, far_values, csi_values, model_names, threshold):
    """
    Create a fit diagram showing model performance with POD vs FAR.

    The diagram shows:
    - X-axis: POD (Probability of Detection) - range [0, 1]
    - Y-axis: FAR (False Alarm Rate) - range [0, 1]
    - Each model plotted with distinct color and marker

    Args:
        pod_values (list): POD values for each model
        far_values (list): FAR values for each model
        csi_values (list): CSI values for each model
        model_names (list): Names of the models
        threshold (float): Precipitation threshold (mm/h)

    Returns:
        matplotlib.figure.Figure: The fit diagram figure
    """
    # Define available colors (cycling through if more models than colors)
    available_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'brown', 'pink', 'olive']

    # Define available markers (cycling through if more models than markers)
    available_markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'd', 'P', 'X']

    # Create a white background plot (match CSI plot height)
    fig, ax = plt.subplots(figsize=(6, 4))

    # Set white background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')

    # Create meshgrid for CSI contours
    pod_grid = np.linspace(0.01, 1, 200)
    far_grid = np.linspace(0, 0.99, 200)
    POD_mesh, FAR_mesh = np.meshgrid(pod_grid, far_grid)

    # Compute CSI for entire grid
    CSI_mesh = compute_csi_from_pod_far(POD_mesh, FAR_mesh)

    # Define CSI contour levels
    csi_levels = np.arange(0.1, 1.0, 0.1)

    # Create filled contours (background shading)
    contourf = ax.contourf(POD_mesh, FAR_mesh, CSI_mesh, levels=20, cmap='Blues', alpha=0.3, zorder=1)

    # Create contour lines for specific CSI values
    contours = ax.contour(POD_mesh, FAR_mesh, CSI_mesh, levels=csi_levels, colors='gray',
                          linewidths=0.7, alpha=0.5, zorder=2)

    # Add labels to contour lines
    ax.clabel(contours, inline=True, fontsize=7, fmt='CSI=%.1f')

    # Draw grid lines
    ax.grid(True, linestyle='--', linewidth=0.4, alpha=0.3, zorder=3)

    # Set axis limits to go from 0 to 1
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Inverted Y-axis (FAR: lower is better)

    # Add colorbar for CSI
    cbar = fig.colorbar(contourf, ax=ax, pad=0.02)
    cbar.set_label('CSI (Critical Success Index)', rotation=270, labelpad=20, fontsize=10)
    cbar.ax.tick_params(labelsize=8)

    # Set labels for axes and adjust their font size
    ax.set_xlabel("POD (Probability of Detection)", fontsize=10, fontweight='bold')
    ax.set_ylabel("FAR (False Alarm Rate)", fontsize=10, fontweight='bold')

    # Set aspect ratio and grid intervals
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(labelsize=8)

    # Plot each model with dynamic color and marker assignment
    for idx, (model_name, pod, far, csi) in enumerate(zip(model_names, pod_values, far_values, csi_values)):
        # Assign color and marker based on index (cycling through lists)
        color = available_colors[idx % len(available_colors)]
        marker = available_markers[idx % len(available_markers)]

        ax.scatter(
            pod,
            far,
            label=f"{model_name} (CSI={csi:.3f})",
            color=color,
            s=80,  # Reduced from 150
            zorder=5,
            alpha=0.9,
            marker=marker,
            edgecolors='black',
            linewidths=1.0  # Reduced from 1.5
        )

    # Create legend with 2 rows
    handles, labels = plt.gca().get_legend_handles_labels()

    # Calculate number of columns to get 2 rows
    num_models = len(model_names)
    ncol = (num_models + 1) // 2  # Divide by 2, rounding up

    legend = plt.legend(
        handles,
        labels,
        loc="lower left",
        fontsize="xx-small",
        ncol=ncol,
        framealpha=0.9
    )

    plt.tight_layout()

    return fig