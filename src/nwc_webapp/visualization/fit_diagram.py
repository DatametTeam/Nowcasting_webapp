"""
Fit diagram visualization for CSI Analysis.

Adapted from fit_diagrams.py create_fit_diagram_short() to work with POD, FAR, and CSI metrics.
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap


def create_performance_fit_diagram(pod_values, far_values, csi_values, model_names, threshold):
    """
    Create a fit diagram showing model performance with POD vs FAR.

    The diagram shows:
    - X-axis: POD (Probability of Detection) - range [0, 1]
    - Y-axis: FAR (False Alarm Rate) - range [0, 1]
    - Colorbar: CSI (Critical Success Index) - shown via colormap
    - Background: Concentric circles representing performance levels

    Args:
        pod_values (list): POD values for each model
        far_values (list): FAR values for each model
        csi_values (list): CSI values for each model
        model_names (list): Names of the models
        threshold (float): Precipitation threshold (mm/h)

    Returns:
        matplotlib.figure.Figure: The fit diagram figure
    """
    # Define model colors (can be extended)
    method_colors = {
        "ConvLSTM": "red",
        "ED_ConvLSTM": "orange",
        "DynamicUnet": "green",
        "pysteps": "blue",
        "Test": "purple",
    }

    # Create a white background plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter([], [], c="white")  # Empty scatter plot for a white background

    # Define the number of circles and colormap
    num_circles = 10
    # Define custom colors for the discrete colormap
    blues_cmap = plt.get_cmap("Blues")
    # Extract colors from the "Blues" colormap and select specific levels
    colors = [blues_cmap(i / 10.0) for i in range(10)]

    # Create a custom discrete colormap with specified colors and levels
    levels = np.arange(0.0, 1.1, 0.1)
    cmap = ListedColormap(colors)

    # Create quarter-circle shapes with the original blue colormap and reduced alpha
    # Center at (1, 0) = perfect performance (POD=1, FAR=0)
    for i in range(num_circles, 0, -1):  # Start from the top
        radius = 0.1 * i  # Adjust radius for each circle
        alpha = 0.7  # Reduced alpha
        circle = plt.Circle(
            (1, 0), radius, color=cmap((num_circles - i) / num_circles, alpha=alpha), fill=True, zorder=2
        )
        plt.gca().add_patch(circle)  # Add the circle to the plot

    # Manually draw grey grid lines with a higher zorder value
    for x in np.arange(0, 1.1, 0.1):
        if x <= 0.9:  # Draw grid lines only in the upper-right quadrant
            plt.axvline(x, color="gray", linestyle="--", linewidth=0.5, zorder=3)
            plt.axhline(x, color="gray", linestyle="--", linewidth=0.5, zorder=3)

    # Set axis limits to go from 0 to 1
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)  # Inverted Y-axis (FAR: lower is better)

    # Add a white diagonal line from (1, 0) to (0, 1)
    ax.plot([1, 0], [0, 1], color="white", linewidth=1, alpha=0.5)

    # Set labels for axes and adjust their font size
    ax.set_xlabel("POD (Probability of Detection)", fontsize=10)
    ax.set_ylabel("FAR (False Alarm Rate)", fontsize=10)

    # Add a colorbar with updated label position and text size
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ticks=levels, ax=ax)
    cbar.set_label("CSI (Critical Success Index)", fontsize=10)
    cbar.ax.yaxis.set_label_position("left")  # Move label to the left side
    cbar.ax.set_position([0.8, 0.005, 0.029, 0.98])  # Adjust the position of the colorbar

    # Set the font size of colorbar labels to match the axis labels
    cbar.ax.tick_params(labelsize=10)

    # Set colorbar ticks every 0.1 with small squares to separate colors and add labels
    cbar.set_ticks(np.arange(0.0, 1.1, 0.1))
    cbar.set_ticklabels(["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])

    # Show the plot
    plt.gca().set_aspect("equal", adjustable="box")  # Make the plot aspect ratio equal

    # Set grid intervals every 0.1 units
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=8)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=8)

    # Define markers for models
    markers = {
        "ConvLSTM": "o",  # Circle
        "ED_ConvLSTM": "D",  # Diamond
        "DynamicUnet": "P",  # Plus (Filled plus)
        "pysteps": "v",  # Downward Triangle
        "Test": "s",  # Square
    }

    # Plot each model
    for model_name, pod, far, csi in zip(model_names, pod_values, far_values, csi_values):
        # Get color and marker (use defaults if model not in predefined dict)
        color = method_colors.get(model_name, "black")
        marker = markers.get(model_name, "o")

        plt.scatter(
            pod,
            far,
            label=model_name,
            color=color,
            s=100,
            zorder=5,
            alpha=0.8,
            marker=marker,
        )

    # Create legend
    handles, labels = plt.gca().get_legend_handles_labels()

    legend = plt.legend(
        handles,
        labels,
        loc="lower left",
        fontsize="xx-small",
        ncol=min(len(model_names), 6),
    )
    plt.setp(legend.get_title(), fontsize="small")

    plt.title(f"Fit Diagram @ {threshold} mm/h")
    plt.tight_layout()

    return fig