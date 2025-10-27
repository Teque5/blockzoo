"""Visualization utilities for BlockZoo framework.

This module provides functions to visualize benchmark results and generate
plots for analyzing block performance across different positions.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd


def load_results(csv_path: str) -> pd.DataFrame:
    """Load results CSV file and perform basic validation.

    Parameters
    ----------
    csv_path : str
        Path to the results CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded and validated results dataframe.

    Raises
    ------
    FileNotFoundError
        If the CSV file doesn't exist.
    ValueError
        If required columns are missing.
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Results file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Check for required columns
    required_cols = ["val_acc", "position", "block"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter out rows with missing data for core columns
    df = df.dropna(subset=required_cols)

    if len(df) == 0:
        raise ValueError("No valid data rows found in CSV")

    return df


def get_position_styles():
    """Get consistent position styling."""
    return {
        "early": {"marker": "o", "linestyle": "-", "markersize": 8},
        "mid": {"marker": "s", "linestyle": "--", "markersize": 8},
        "late": {"marker": "^", "linestyle": "-.", "markersize": 8},
    }


def get_block_colors(blocks):
    """Get consistent color mapping for blocks."""
    colors = plt.cm.tab10(range(len(blocks)))
    return dict(zip(blocks, colors))


def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, x_label: str, y_label: str, title: str, output_path: str) -> None:
    """Create a generic accuracy scatter plot.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    x_col : str
        Column name for x-axis data
    y_col : str
        Column name for y-axis data (should be 'val_acc')
    x_label : str
        Label for x-axis
    y_label : str
        Label for y-axis
    title : str
        Plot title
    output_path : str
        Path to save the plot
    """
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=(12, 8))

    # Check if required columns exist
    if x_col not in df.columns:
        print(f"Warning: Column '{x_col}' not found, skipping plot")
        return

    # Filter out rows with missing data for this plot
    plot_df = df.dropna(subset=[x_col, y_col])
    if len(plot_df) == 0:
        print(f"Warning: No valid data for {x_col} vs {y_col} plot")
        return

    position_styles = get_position_styles()
    blocks = plot_df["block"].unique()
    block_colors = get_block_colors(blocks)

    # Create scatter plot for each block-position combination
    for block in blocks:
        block_data = plot_df[plot_df["block"] == block]

        for position in ["early", "mid", "late"]:
            pos_data = block_data[block_data["position"] == position]

            if len(pos_data) == 0:
                continue

            style = position_styles[position]
            color = block_colors[block]

            # Plot points
            ax.scatter(
                pos_data[x_col],
                pos_data[y_col],
                color=color,
                marker=style["marker"],
                s=style["markersize"] ** 2,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
                label=f"{block} ({position})",
            )

    # Connect points for the same block across positions with lines
    for block in blocks:
        block_data = plot_df[plot_df["block"] == block]

        # Group by position and get mean values for line plotting
        pos_means = block_data.groupby("position").agg({x_col: "mean", y_col: "mean"}).reindex(["early", "mid", "late"])

        # Only plot line if we have data for multiple positions
        valid_positions = pos_means.dropna()
        if len(valid_positions) >= 2:
            ax.plot(valid_positions[x_col], valid_positions[y_col], color=block_colors[block], alpha=0.6, linewidth=2, linestyle="-")

    # Customize the plot
    ax.set_ylim(0.7, 1)
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Create compact combined legend
    legend_elements = []

    # Position markers section
    for pos, style in position_styles.items():
        legend_elements.append(plt.Line2D([0], [0], marker=style["marker"], color="black", linestyle="None", markersize=8, label=f"{pos.title()} Position"))

    # Block colors section
    for block, color in block_colors.items():
        legend_elements.append(plt.Line2D([0], [0], marker="o", color=color, linestyle="None", markersize=8, label=block))

    # Create single legend with two columns
    legend = ax.legend(
        handles=legend_elements,
        ncol=2,
        loc="lower right",
    )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to: {output_path}")


def create_all_plots(df: pd.DataFrame, output_dir: str = ".") -> None:
    """Create all four accuracy comparison plots.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe
    output_dir : str
        Directory to save plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Plot configurations: (column, label, filename)
    plot_configs = [
        # ("throughput", "Throughput (images/second)", "accuracy_vs_throughput.png"),
        ("params_total", "Total Parameters", "accuracy_vs_params.png"),
        ("memory_mb", "Memory Usage (MB)", "accuracy_vs_memory.png"),
        ("latency_ms", "Latency (ms)", "accuracy_vs_latency.png"),
    ]

    for x_col, x_label, filename in plot_configs:
        title = f"Block Performance: Accuracy vs {x_label.split('(')[0].strip()}"
        output_file = output_path / filename

        create_scatter_plot(df=df, x_col=x_col, y_col="val_acc", x_label=x_label, y_label="Validation Accuracy", title=title, output_path=str(output_file))


def main():
    """Main entry point for visualization CLI."""
    parser = argparse.ArgumentParser(description="Generate BlockZoo benchmark visualization plots", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("csv_path", help="Path to the results CSV file")
    parser.add_argument("--output-dir", "-o", default="plots", help="Output directory for plots (default: current directory)")

    args = parser.parse_args()

    try:
        # Load results
        print(f"Loading results from: {args.csv_path}")
        df = load_results(args.csv_path)
        print(f"Loaded {len(df)} experiments")

        # Create all plots
        print("Generating visualization plots...")
        create_all_plots(df, args.output_dir)

        print("\nGenerated plots:")
        print("  • accuracy_vs_throughput.png - Accuracy vs Throughput")
        print("  • accuracy_vs_params.png - Accuracy vs Parameters")
        print("  • accuracy_vs_memory.png - Accuracy vs Memory Usage")
        print("  • accuracy_vs_latency.png - Accuracy vs Latency")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
