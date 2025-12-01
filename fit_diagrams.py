import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

from ._utils import dpi


def create_fit_diagram(mean_result_df, save_folder):
    methods_selected = ["XGB", "LGB", "RF", "ANN", "MaxLike", "MaxFit", "MaxEff", "R$_{Comb}$_200"]
    order = [0, 2, 6, 3, 1, 4, 7, 5]
    methods_to_be_used = [name for name in methods_selected if name in mean_result_df["Method"].values]
    ordered_methods = [methods_selected[i] for i in order if methods_selected[i] in methods_to_be_used]
    method_colors = {
        "GB": "purple",
        "ANN_ENSEMBLE": "cyan",
        "ANN": "blue",
        "XGB": "red",
        "RF": "orange",
        "LGB": "green",
        "MaxEff": "black",
        "MaxFit": "#777777",
        "MaxLike": "#666666",
        "R$_{Comb}$_200": "#444444",
        "R_op": "#333333",
    }

    fig, ax = plt.subplots(figsize=(8, 6))  # create fig and ax explicitly

    # Create a white background
    ax.scatter([], [], c="white")

    # Define the number of circles and colormap (original blue)
    num_circles = 10
    # Define custom colors for the discrete colormap
    blues_cmap = plt.get_cmap("Blues")
    # Extract colors from the "Blues" colormap and select specific levels
    colors = [blues_cmap(i / 10.0) for i in range(10)]

    # Create a custom discrete colormap with specified colors and levels
    levels = np.arange(0.0, 1.1, 0.1)
    cmap = ListedColormap(colors)

    # Create quarter-circle shapes with the original blue colormap and reduced alpha
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
            plt.axvline(x, color="gray", linestyle="--", linewidth=0.5, zorder=3)  # Increased zorder
            plt.axhline(x, color="gray", linestyle="--", linewidth=0.5, zorder=3)  # Increased zorder

    # Set axis limits to go from 0 to 1
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    # Add a white diagonal line from (1, 0) to (0, 1)
    ax.plot([1, 0], [0, 1], color="white", linewidth=1, alpha=0.5)

    # Set labels for axes and adjust their font size
    ax.set_xlabel("R²", fontsize=10)
    ax.set_ylabel("Abs. Rel. Error", fontsize=10)

    # Add a colorbar with updated label position and text size
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ticks=levels, ax=ax)
    cbar.set_label("Goodness of Fit", fontsize=10)  # Update label text size
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

    markers = {
        "MaxEff": "s",  # Circle
        "MaxFit": "X",  # Cross
        "MaxLike": "^",  # Triangle
    }

    for category in np.unique(mean_result_df["Method"]):
        if category in methods_to_be_used:
            marker = markers.get(category, "o")  # Default to circle if not specified

            plt.scatter(
                mean_result_df[mean_result_df["Method"] == category]["R²"],
                mean_result_df[mean_result_df["Method"] == category]["Rel. Err."],
                label=category,
                color=method_colors[category],
                s=100,
                zorder=5,
                alpha=0.8,
                marker=marker,
            )

    # legend = plt.legend(title='Method', loc='upper left', fontsize='xx-small', ncol=4)
    handles, labels = plt.gca().get_legend_handles_labels()
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    idxs = [label_to_idx[m] for m in ordered_methods if m in label_to_idx]

    legend = plt.legend(
        [handles[idx] for idx in idxs], [labels[idx] for idx in idxs], loc="lower left", fontsize="xx-small", ncol=4
    )
    # legend = plt.legend(loc='lower left', fontsize='xx-small', ncol=4)
    plt.setp(legend.get_title(), fontsize="small")  # Set the legend title font size

    plt.title("Fit Diagram")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_folder, "results", "fit_diagram.png"), bbox_inches="tight", dpi=dpi)
    plt.close()


def create_fit_diagram_short(mean_result_df, save_folder):
    methods_to_be_used = ["XGB", "LGB", "RF", "ANN", "R$_{Comb}$_200", "MaxEff"]  # 'MaxLike', 'MaxFit',
    # 'MaxEff', # 'R$_{Comb}$_200']

    method_colors = {
        "GB": "purple",
        "ANN_ENSEMBLE": "cyan",
        "ANN": "blue",
        "XGB": "red",
        "RF": "orange",
        "LGB": "green",
        "MaxEff": "black",
        "MaxFit": "#777777",
        "MaxLike": "#666666",
        "R$_{Comb}$_200": "#444444",
        "R$_{Comb}$_300": "#666666",
        "R_op": "#333333",
    }

    # Create a white background plot
    fig, ax = plt.subplots(figsize=(8, 6))  # create fig and ax explicitly
    ax.scatter([], [], c="white")  # Empty scatter plot for a white background

    # Define the number of circles and colormap (original blue)
    num_circles = 10
    # Define custom colors for the discrete colormap
    blues_cmap = plt.get_cmap("Blues")
    # Extract colors from the "Blues" colormap and select specific levels
    colors = [blues_cmap(i / 10.0) for i in range(10)]

    # Create a custom discrete colormap with specified colors and levels
    levels = np.arange(0.0, 1.1, 0.1)
    cmap = ListedColormap(colors)

    # Create quarter-circle shapes with the original blue colormap and reduced alpha
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
            plt.axvline(x, color="gray", linestyle="--", linewidth=0.5, zorder=3)  # Increased zorder
            plt.axhline(x, color="gray", linestyle="--", linewidth=0.5, zorder=3)  # Increased zorder

    # Set axis limits to go from 0 to 1
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)

    # Add a white diagonal line from (1, 0) to (0, 1)
    ax.plot([1, 0], [0, 1], color="white", linewidth=1, alpha=0.5)

    # Set labels for axes and adjust their font size
    ax.set_xlabel("R²", fontsize=10)
    ax.set_ylabel("Abs. Rel. Error", fontsize=10)

    # Add a colorbar with updated label position and text size
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ticks=levels, ax=ax)
    cbar.set_label("Goodness of Fit", fontsize=10)  # Update label text size
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

    markers = {
        "XGB": "o",  # Circle
        "LGB": "D",  # Diamond
        "RF": "P",  # Plus (Filled plus)
        "ANN": "v",  # Downward Triangle
        "R$_{Comb}$_200": "s",  # Square
    }

    for category in np.unique(mean_result_df["Method"]):
        if category in methods_to_be_used:
            marker = markers.get(category, "o")  # Default to circle if not specified

            plt.scatter(
                mean_result_df[mean_result_df["Method"] == category]["R²"],
                mean_result_df[mean_result_df["Method"] == category]["Rel. Err."],
                label=category,
                color=method_colors[category],
                s=100,
                zorder=5,
                alpha=0.8,
                marker=marker,
            )

    # legend = plt.legend(title='Method', loc='upper left', fontsize='xx-small', ncol=4)
    handles, labels = plt.gca().get_legend_handles_labels()

    labels = ["R_fit" if label == "MaxEff" else label for label in labels]

    # order = [0, 2, 6, 3, 1, 4, 7, 5]
    order = [3, 2, 4, 0, 1, 5]
    # Filter order to only include valid indices
    valid_order = [idx for idx in order if idx < len(handles)]

    legend = plt.legend(
        [handles[idx] for idx in valid_order],
        [labels[idx] for idx in valid_order],
        loc="lower left",
        fontsize="xx-small",
        ncol=6,
    )
    # legend = plt.legend(loc='lower left', fontsize='xx-small', ncol=4)
    plt.setp(legend.get_title(), fontsize="small")  # Set the legend title font size

    plt.title("Fit Diagram")
    plt.tight_layout()
    # plt.show()
    plt.savefig(os.path.join(save_folder, "results", "fit_diagram_short.png"), bbox_inches="tight", dpi=dpi)
    plt.close()


def create_station_fit_diagram(mean_result_df, save_folder):
    from ._utils import generate_colors

    os.makedirs(os.path.join(save_folder, "graph_folder", "fit_diagrams"), exist_ok=True)
    stations_to_be_used = mean_result_df["radar_station"].unique()
    station_colors = generate_colors(len(stations_to_be_used))

    for method in mean_result_df["Method"].unique():

        # Create a white background plot
        plt.figure(figsize=(8, 6))
        plt.scatter([], [], c="white")  # Empty scatter plot for a white background

        # Define the number of circles and colormap (original blue)
        num_circles = 10
        # Define custom colors for the discrete colormap
        blues_cmap = plt.get_cmap("Blues")
        # Extract colors from the "Blues" colormap and select specific levels
        colors = [blues_cmap(i / 10.0) for i in range(10)]

        # Create a custom discrete colormap with specified colors and levels
        levels = np.arange(0.0, 1.1, 0.1)
        cmap = ListedColormap(colors)

        # Create quarter-circle shapes with the original blue colormap and reduced alpha
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
                plt.axvline(x, color="gray", linestyle="--", linewidth=0.5, zorder=3)  # Increased zorder
                plt.axhline(x, color="gray", linestyle="--", linewidth=0.5, zorder=3)  # Increased zorder

        # Set axis limits to go from 0 to 1
        plt.xlim(0, 1)
        plt.ylim(1, 0)

        # Add a white diagonal line from (1, 0) to (0, 1)
        plt.plot([1, 0], [0, 1], color="white", linewidth=1, alpha=0.5)

        # Set labels for axes and adjust their font size
        plt.xlabel("R²", fontsize=10)
        plt.ylabel("Abs. Rel. Error", fontsize=10)

        # Add a colorbar with updated label position and text size
        cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=plt.gca(), ticks=levels)
        cbar.set_label("Goodness of Fit", fontsize=10)  # Update label text size
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

        symbols = ["o", "s", "^", "D", "X", "P"]
        for i, station in enumerate(stations_to_be_used):
            plt.scatter(
                mean_result_df[(mean_result_df["radar_station"] == station) & (mean_result_df["Method"] == method)][
                    "R²"
                ],
                mean_result_df[(mean_result_df["radar_station"] == station) & (mean_result_df["Method"] == method)][
                    "Rel. Err."
                ],
                label=station,
                color=station_colors[i],
                marker=symbols[i % len(symbols)],
                s=50,
                zorder=5,
                alpha=0.8,
            )

        # legend = plt.legend(title='Method', loc='upper left', fontsize='xx-small', ncol=4)
        # handles, labels = plt.gca().get_legend_handles_labels()
        # order = [0, 2, 1, 3, 5, 4, 6]

        legend = plt.legend(loc="lower left", fontsize=7, ncol=4)
        # legend = plt.legend(loc='lower left', fontsize='xx-small', ncol=4)
        plt.setp(legend.get_title(), fontsize="small")  # Set the legend title font size

        if method == "MaxEff":
            method = "R_fit"

        plt.title(f"Fit Diagram for {method}")
        # plt.show()
        plt.savefig(
            os.path.join(save_folder, "graph_folder", "fit_diagrams", f"fit_diagram_{method}.png"),
            bbox_inches="tight",
            dpi=dpi,
        )
        plt.close()


def create_station_fit_diagram_short(mean_result_df, save_folder):
    os.makedirs(os.path.join(save_folder, "graph_folder", "fit_diagrams"), exist_ok=True)
    stations_to_be_used = ["ARMIDDA", "ILMONTE", "PETTINASCURA", "SERANO", "ZOUFPLAN"]

    # Define colors for methods and shapes for stations
    method_colors = {"XGB": "red", "R_op": "blue"}
    station_shapes = {
        "SERANO": "o",  # Circle
        "ZOUFPLAN": "s",  # Square
        "PETTINASCURA": "^",  # Triangle
        "ARMIDDA": "D",  # Diamond
        "ILMONTE": "P",  # Inverted Triangle
    }
    # Create a white background plot
    plt.figure(figsize=(8, 6))
    plt.scatter([], [], c="white")  # Empty scatter plot for a white background

    # Define the number of circles and colormap (original blue)
    num_circles = 10
    blues_cmap = plt.get_cmap("Blues")
    colors = [blues_cmap(i / 10.0) for i in range(10)]  # Extract colors from the "Blues" colormap
    cmap = ListedColormap(colors)
    levels = np.arange(0.0, 1.1, 0.1)

    # Create quarter-circle shapes with the original blue colormap and reduced alpha
    for i in range(num_circles, 0, -1):
        radius = 0.1 * i
        alpha = 0.7
        circle = plt.Circle(
            (1, 0), radius, color=cmap((num_circles - i) / num_circles, alpha=alpha), fill=True, zorder=2
        )
        plt.gca().add_patch(circle)

    # Manually draw grey grid lines
    for x in np.arange(0, 1.1, 0.1):
        if x <= 0.9:
            plt.axvline(x, color="gray", linestyle="--", linewidth=0.5, zorder=3)
            plt.axhline(x, color="gray", linestyle="--", linewidth=0.5, zorder=3)

    # Set axis limits, diagonal line, and labels
    plt.xlim(0, 1)
    plt.ylim(1, 0)
    plt.plot([1, 0], [0, 1], color="white", linewidth=1, alpha=0.5)
    plt.xlabel("R²", fontsize=10)
    plt.ylabel("Abs. Rel. Error", fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=plt.gca(), ticks=levels)
    cbar.set_label("Goodness of Fit", fontsize=10)
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.set_position([0.8, 0.005, 0.029, 0.98])
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks(np.arange(0.0, 1.1, 0.1))
    cbar.set_ticklabels(["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])

    # Set aspect ratio and grid intervals
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=8)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=8)

    armidda_data = (0.35, 0.43)
    ilmonte_data = (0.45, 0.5)

    # Plotting results for both methods
    for method in ["XGB", "R_op"]:
        color = method_colors[method]
        for station in stations_to_be_used:

            station_data = mean_result_df[
                (mean_result_df["radar_station"] == station) & (mean_result_df["Method"] == method)
            ]
            if station_data.empty:
                continue
            plt.scatter(
                station_data["R²"],
                station_data["Rel. Err."],
                color=color,
                marker=station_shapes[station],
                label=f"{method} - {station}",
                s=50,
                zorder=5,
                alpha=0.8,
                edgecolor="black",  # Add black border around markers
                linewidth=0.5,  # Thickness of the border
            )

        # Custom legend elements
        legend_elements_XGB = [
            Line2D([0], [0], marker="D", color="none", markerfacecolor="red", markersize=7, label="ARMIDDA"),
            Line2D([0], [0], marker="P", color="none", markerfacecolor="red", markersize=7, label="ILMONTE"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor="red", markersize=7, label="PETTINASCURA"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="red", markersize=7, label="SERANO"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="red", markersize=7, label="ZOUFPLAN"),
        ]

        legend_elements_Rop = [
            Line2D([0], [0], marker="D", color="none", markerfacecolor="blue", markersize=7, label="ARMIDDA"),
            Line2D([0], [0], marker="P", color="none", markerfacecolor="blue", markersize=7, label="ILMONTE"),
            Line2D([0], [0], marker="^", color="none", markerfacecolor="blue", markersize=7, label="PETTINASCURA"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="blue", markersize=7, label="SERANO"),
            Line2D([0], [0], marker="s", color="none", markerfacecolor="blue", markersize=7, label="ZOUFPLAN"),
        ]

        # Create legend with two columns
        legend = plt.legend(
            handles=legend_elements_XGB + legend_elements_Rop,
            loc="lower left",
            fontsize=7,
            title="XGB                R$_{OP}$     ",
            title_fontsize="small",
            ncol=2,
        )

        # Set plot title and save
        plt.title("Fit Diagram for XGB and R$_{OP}$ Methods")
        plt.savefig(
            os.path.join(save_folder, "graph_folder", "fit_diagrams", "fit_diagram_combined_short.png"),
            bbox_inches="tight",
            dpi=dpi,
        )
        # plt.show()  # Uncomment to display plot interactively


def create_station_fit_diagram_comparison_full(mean_result_df, save_folder):
    def jitter(data, jitter_strength=0.02):
        """Add jitter to the data."""
        return data + np.random.uniform(-jitter_strength, jitter_strength, size=data.shape)

    os.makedirs(os.path.join(save_folder, "graph_folder", "fit_diagrams"), exist_ok=True)
    stations_to_be_used = mean_result_df["radar_station"].unique()

    # Define colors for methods and shapes for stations
    method_colors = {"XGB": "red", "R_op": "blue"}
    available_markers = ["o", "s", "^", "D", "P", "v", ">", "<", "*", "h", "H", "d", "X"]
    station_shapes = {
        station: available_markers[i % len(available_markers)] for i, station in enumerate(stations_to_be_used)
    }

    # Create a white background plot
    plt.figure(figsize=(8, 6))
    plt.scatter([], [], c="white")  # Empty scatter plot for a white background

    # Define the number of circles and colormap (original blue)
    num_circles = 10
    blues_cmap = plt.get_cmap("Blues")
    colors = [blues_cmap(i / 10.0) for i in range(10)]  # Extract colors from the "Blues" colormap
    cmap = ListedColormap(colors)
    levels = np.arange(0.0, 1.1, 0.1)

    # Create quarter-circle shapes with the original blue colormap and reduced alpha
    for i in range(num_circles, 0, -1):
        radius = 0.1 * i
        alpha = 0.7
        circle = plt.Circle(
            (1, 0), radius, color=cmap((num_circles - i) / num_circles, alpha=alpha), fill=True, zorder=2
        )
        plt.gca().add_patch(circle)

    # Manually draw grey grid lines
    for x in np.arange(0, 1.1, 0.1):
        if x <= 0.9:
            plt.axvline(x, color="gray", linestyle="--", linewidth=0.5, zorder=3)
            plt.axhline(x, color="gray", linestyle="--", linewidth=0.5, zorder=3)

    # Set axis limits, diagonal line, and labels
    plt.xlim(0, 1)
    plt.ylim(1, 0)
    plt.plot([1, 0], [0, 1], color="white", linewidth=1, alpha=0.5)
    plt.xlabel("R²", fontsize=10)
    plt.ylabel("Abs. Rel. Error", fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=plt.gca(), ticks=levels)
    cbar.set_label("Goodness of Fit", fontsize=10)
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.set_position([0.8, 0.005, 0.029, 0.98])
    cbar.ax.tick_params(labelsize=10)
    cbar.set_ticks(np.arange(0.0, 1.1, 0.1))
    cbar.set_ticklabels(["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9", "1.0"])

    # Set aspect ratio and grid intervals
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xticks(np.arange(0, 1.1, 0.1), fontsize=8)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=8)

    # Plotting results for both methods
    for method in ["XGB", "R_op"]:
        color = method_colors[method]
        for station in stations_to_be_used:

            station_data = mean_result_df[
                (mean_result_df["radar_station"] == station) & (mean_result_df["Method"] == method)
            ]
            if station_data.empty:
                continue
            plt.scatter(
                jitter(station_data["R²"]),
                jitter(station_data["Rel. Err."]),
                color=color,
                marker=station_shapes[station],
                label=f"{method} - {station}",
                s=25,
                zorder=5,
                alpha=0.8,
                edgecolor="black",  # Add black border around markers
                linewidth=0.5,  # Thickness of the border
            )

        # Custom legend elements
        # Create custom legend elements
        legend_elements_LGB = []
        legend_elements_Rop = []

        for station in stations_to_be_used:
            legend_elements_LGB.append(
                Line2D(
                    [0],
                    [0],
                    marker=station_shapes[station],
                    color="none",
                    markerfacecolor="red",
                    markersize=6,
                    label=f"{station}",
                )
            )
            legend_elements_Rop.append(
                Line2D(
                    [0],
                    [0],
                    marker=station_shapes[station],
                    color="none",
                    markerfacecolor="blue",
                    markersize=6,
                    label=f"{station}",
                )
            )

        # Add a dummy element for balancing the columns if the number of stations is odd
        if len(stations_to_be_used) % 2 != 0:
            legend_elements_LGB.append(Line2D([0], [0], linestyle="none", marker="None", label=""))
            legend_elements_Rop.append(Line2D([0], [0], linestyle="none", marker="None", label=""))

        # Create legend with four columns (2 for LGB and 2 for R$_{OP}$)
        plt.legend(
            handles=legend_elements_LGB + legend_elements_Rop,
            loc="lower left",
            fontsize=6,
            ncol=4,
            title="  XGB                           R$_{OP}$           ",
            title_fontsize="small",
            handletextpad=0.5,
            columnspacing=1.1,
        )

        # Set plot title and save
        plt.title("Fit Diagram for XGB and R$_{OP}$ Methods")
        plt.savefig(
            os.path.join(save_folder, "graph_folder", "fit_diagrams", "fit_diagram_combined_full.png"),
            bbox_inches="tight",
            dpi=dpi,
        )
        # plt.show()  # Uncomment to display plot interactively
