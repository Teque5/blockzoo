"""Integration tests for BlockZoo framework."""

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from blockzoo.profiler import profile_block_in_scaffold
from blockzoo.utils import append_results, load_results


class TestIntegration(unittest.TestCase):
    """Integration tests for BlockZoo framework."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.temp_csv = self.temp_dir / "integration_results.csv"

    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_end_to_end_profiling(self):
        """Test end-to-end profiling workflow."""
        # profile a block
        profile = profile_block_in_scaffold("ResNetBasicBlock", position="early")

        # save results
        append_results(self.temp_csv, profile)

        # load and verify results
        df = load_results(str(self.temp_csv))
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["position"], "early")

    def test_multiple_positions(self):
        """Test profiling across multiple positions."""
        positions = ["early", "mid", "late"]

        for position in positions:
            profile = profile_block_in_scaffold("ResNetBasicBlock", position=position)
            profile["test_run"] = f"position_{position}"
            append_results(self.temp_csv, profile)

        # load and verify results
        df = load_results(str(self.temp_csv))
        self.assertEqual(len(df), 3)

        # check that all positions are represented
        positions_in_results = set(df["position"].values)
        self.assertEqual(positions_in_results, set(positions))


if __name__ == "__main__":
    unittest.main()
