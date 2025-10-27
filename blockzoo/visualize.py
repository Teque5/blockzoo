"""Visualization utilities for BlockZoo framework.

This module provides functions to visualize benchmark results and generate
plots for analyzing block performance across different positions.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.markers import MarkerStyle


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
    required_cols = ["val_acc", "throughput", "position", "block"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Filter out rows with missing data
    df = df.dropna(subset=required_cols)

    if len(df) == 0:
        raise ValueError("No valid data rows found in CSV")

    return df


def create_accuracy_throughput_plot(df: pd.DataFrame, output_path: Optional[str] = None) -> None:
    """Create accuracy vs throughput scatter plot.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe with columns: val_acc, throughput, position, block
    output_path : str, optional
        Path to save the plot. If None, displays the plot.
    """
    # Set up the plot style
    plt.style.use("seaborn-v0_8")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Define position styles (markers and line styles)
    position_styles = {
        "early": {"marker": "o", "linestyle": "-", "markersize": 8},
        "mid": {"marker": "s", "linestyle": "--", "markersize": 8},
        "late": {"marker": "^", "linestyle": "-.", "markersize": 8},
    }

    # Get unique blocks and assign colors
    blocks = df["block"].unique()
    colors = plt.cm.tab10(range(len(blocks)))
    block_colors = dict(zip(blocks, colors))

    # Create scatter plot for each block-position combination
    for block in blocks:
        block_data = df[df["block"] == block]

        for position in ["early", "mid", "late"]:
            pos_data = block_data[block_data["position"] == position]

            if len(pos_data) == 0:
                continue

            style = position_styles[position]
            color = block_colors[block]

            # Plot points
            ax.scatter(
                pos_data["throughput"],
                pos_data["val_acc"],
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
        block_data = df[df["block"] == block]

        # Group by position and get mean values for line plotting
        pos_means = block_data.groupby("position").agg({"throughput": "mean", "val_acc": "mean"}).reindex(["early", "mid", "late"])

        # Only plot line if we have data for multiple positions
        valid_positions = pos_means.dropna()
        if len(valid_positions) >= 2:
            ax.plot(valid_positions["throughput"], valid_positions["val_acc"], color=block_colors[block], alpha=0.6, linewidth=2, linestyle="-")

    # Customize the plot
    ax.set_xlabel("Throughput (images/second)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Validation Accuracy", fontsize=12, fontweight="bold")
    ax.set_title("Block Performance: Accuracy vs Throughput by Position", fontsize=14, fontweight="bold", pad=20)

    # Add grid
    ax.grid(True, alpha=0.3)

    # Create custom legend
    # First, create legend for positions (markers)
    position_legend_elements = []
    for pos, style in position_styles.items():
        position_legend_elements.append(
            plt.Line2D([0], [0], marker=style["marker"], color="black", linestyle="None", markersize=8, label=f"{pos.title()} Position")
        )

    # Create legend for blocks (colors)
    block_legend_elements = []
    for block, color in block_colors.items():
        block_legend_elements.append(plt.Line2D([0], [0], marker="o", color=color, linestyle="None", markersize=8, label=block))

    # Create two separate legends
    pos_legend = ax.legend(handles=position_legend_elements, title="Position", loc="upper left", bbox_to_anchor=(0.02, 0.98))
    pos_legend.get_title().set_fontweight("bold")

    block_legend = ax.legend(handles=block_legend_elements, title="Block Type", loc="lower right", bbox_to_anchor=(0.98, 0.02))
    block_legend.get_title().set_fontweight("bold")

    # Add the position legend back (since legend() replaces the previous one)
    ax.add_artist(pos_legend)

    # Adjust layout
    plt.tight_layout()

    # Save or display
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def print_summary_stats(df: pd.DataFrame) -> None:
    """Print summary statistics for the results.

    Parameters
    ----------
    df : pd.DataFrame
        Results dataframe.
    """
    print("\n" + "=" * 60)
    print("BLOCKZOO RESULTS SUMMARY")
    print("=" * 60)

    print(f"Total experiments: {len(df)}")
    print(f"Unique blocks: {len(df['block'].unique())}")
    print(f"Positions tested: {sorted(df['position'].unique())}")

    print(f"\nAccuracy range: {df['val_acc'].min():.3f} - {df['val_acc'].max():.3f}")
    print(f"Throughput range: {df['throughput'].min():.1f} - {df['throughput'].max():.1f} img/s")

    # Best performing configurations
    print(f"\nTop 5 by Accuracy:")
    top_acc = df.nlargest(5, "val_acc")[["block", "position", "val_acc", "throughput"]]
    for _, row in top_acc.iterrows():
        print(f"  {row['block']} ({row['position']}): {row['val_acc']:.3f} acc, {row['throughput']:.0f} img/s")

    print(f"\nTop 5 by Throughput:")
    top_throughput = df.nlargest(5, "throughput")[["block", "position", "val_acc", "throughput"]]
    for _, row in top_throughput.iterrows():
        print(f"  {row['block']} ({row['position']}): {row['throughput']:.0f} img/s, {row['val_acc']:.3f} acc")

    # Position analysis
    print(f"\nPerformance by Position:")
    pos_stats = df.groupby("position").agg({"val_acc": ["mean", "std"], "throughput": ["mean", "std"]}).round(3)

    for pos in ["early", "mid", "late"]:
        if pos in pos_stats.index:
            acc_mean = pos_stats.loc[pos, ("val_acc", "mean")]
            acc_std = pos_stats.loc[pos, ("val_acc", "std")]
            thr_mean = pos_stats.loc[pos, ("throughput", "mean")]
            thr_std = pos_stats.loc[pos, ("throughput", "std")]
            print(f"  {pos.title()}: {acc_mean:.3f}±{acc_std:.3f} acc, {thr_mean:.0f}±{thr_std:.0f} img/s")


def main():
    """Main entry point for visualization CLI."""
    parser = argparse.ArgumentParser(description="Visualize BlockZoo benchmark results", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("csv_path", help="Path to the results CSV file")
    parser.add_argument("--output", "-o", help="Output path for the plot (PNG/PDF). If not specified, displays the plot.")
    parser.add_argument("--summary", "-s", action="store_true", help="Print summary statistics")
    parser.add_argument("--figsize", nargs=2, type=float, default=[12, 8], metavar=("WIDTH", "HEIGHT"), help="Figure size in inches")

    args = parser.parse_args()

    try:
        # Load results
        print(f"Loading results from: {args.csv_path}")
        df = load_results(args.csv_path)

        # Print summary if requested
        if args.summary:
            print_summary_stats(df)

        # Create visualization
        print("Generating accuracy vs throughput plot...")
        create_accuracy_throughput_plot(df, args.output)

        if not args.output:
            print("Close the plot window to continue...")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
