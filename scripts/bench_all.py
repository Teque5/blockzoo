#!/usr/bin/env python3
"""
Simple BlockZoo benchmark script.

Runs blockzoo-train on all compatible block types at all positions.
"""

import subprocess
import sys
from pathlib import Path

# use blockzoo.wrappers for proper timm block integration
from blockzoo.wrappers import list_available_blocks


def main():
    """Run benchmark on all blocks at all positions."""
    blocks = list_available_blocks()
    positions = ["early", "mid", "late"]
    # show available blocks from wrappers
    print("Available blocks in blockzoo.wrappers:")
    for block in list_available_blocks():
        print(f"  - {block}")
    print()

    # ensure results directory exists
    Path("results").mkdir(exist_ok=True)

    total_experiments = len(blocks) * len(positions)
    current = 0

    print(f"Running {total_experiments} experiments...")

    for block in blocks:
        for position in positions:
            current += 1
            print(f"\n[{current}/{total_experiments}] {block} @ {position}")

            cmd = [sys.executable, "-m", "blockzoo.train", f"blockzoo.wrappers.{block}", "--position", position, "--benchmark"]  # use wrappers module

            try:
                subprocess.run(cmd, check=True)
                print("✅ Success")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed: {e}")
                continue
            except Exception as e:
                print(f"💥 Error: {e}")
                continue

    print(f"\nDone! Results saved to results/results.csv")


if __name__ == "__main__":
    main()
