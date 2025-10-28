"""Visualization utilities for BlockZoo framework.

This module provides functions to visualize benchmark results and generate
plots for analyzing block performance across different positions.
"""

import argparse
import datetime
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


def get_position_sizes():
    """Get consistent position sizing for markers."""
    return {
        "early": 6,  # small
        "mid": 10,  # medium
        "late": 14,  # large
    }


def get_block_styles(blocks):
    """Get consistent marker styles for blocks."""
    # Different marker shapes for different blocks
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "+", "x"]
    colors = plt.cm.tab10(range(len(blocks)))

    styles = {}
    for i, block in enumerate(blocks):
        styles[block] = {"marker": markers[i % len(markers)], "color": colors[i]}
    return styles


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

    position_sizes = get_position_sizes()
    blocks = plot_df["block"].unique()
    block_styles = get_block_styles(blocks)

    # Create scatter plot for each block-position combination
    for block in blocks:
        block_data = plot_df[plot_df["block"] == block]
        block_style = block_styles[block]

        for position in ["early", "mid", "late"]:
            pos_data = block_data[block_data["position"] == position]

            if len(pos_data) == 0:
                continue

            markersize = position_sizes[position]

            # Plot points
            ax.scatter(
                pos_data[x_col],
                pos_data[y_col],
                color=block_style["color"],
                marker=block_style["marker"],
                s=markersize**2,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.5,
                label=f"{block} ({position})",
            )

    # Connect points for the same block across positions with lines
    for block in blocks:
        block_data = plot_df[plot_df["block"] == block]
        block_style = block_styles[block]

        # Group by position and get mean values for line plotting
        pos_means = block_data.groupby("position").agg({x_col: "mean", y_col: "mean"}).reindex(["early", "mid", "late"])

        # Only plot line if we have data for multiple positions
        valid_positions = pos_means.dropna()
        if len(valid_positions) >= 2:
            ax.plot(valid_positions[x_col], valid_positions[y_col], color=block_style["color"], alpha=0.6, linewidth=2, linestyle="-")

    # Customize the plot
    ax.set_ylim(0.5, 0.95)
    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Create compact combined legend
    legend_elements = []

    # Position size section
    for pos, size in position_sizes.items():
        legend_elements.append(plt.Line2D([0], [0], marker="o", color="gray", linestyle="None", markersize=size, label=f"{pos.title()} Position"))

    # Block marker shapes section
    for block, style in block_styles.items():
        legend_elements.append(plt.Line2D([0], [0], marker=style["marker"], color=style["color"], linestyle="None", markersize=8, label=block))

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
    today = datetime.date.today().isoformat()
    # Plot configurations: (column, label, filename)
    plot_configs = [
        ("flops", "FLOPs (Floating Point Operations)", f"{today}_accuracy-vs-flops.png"),
        ("latency_ms", "Inference Latency (ms)", f"{today}_accuracy-vs-latency.png"),
        ("memory_mb", "Memory Usage (MB)", f"{today}_accuracy-vs-memory.png"),
    ]

    for x_col, x_label, filename in plot_configs:
        title = f"BlockZoo: Accuracy vs {x_label.split('(')[0].strip()}"
        output_file = output_path / filename

        create_scatter_plot(df=df, x_col=x_col, y_col="val_acc", x_label=x_label, y_label="CIFAR10 Accuracy", title=title, output_path=str(output_file))


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

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
