"""Tests for blockzoo.utils module."""

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from blockzoo.utils import append_results, format_bytes, load_results


class TestUtils(unittest.TestCase):
    """Test cases for utility functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.temp_csv = self.temp_dir / "test_results.csv"

    def tearDown(self):
        """Clean up test fixtures."""
        if hasattr(self, "temp_dir") and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_format_bytes(self):
        """Test format_bytes function."""
        test_cases = [
            (0, "0 B"),
            (1024, "1.0 KB"),
            (1048576, "1.0 MB"),
            (1073741824, "1.0 GB"),
            (1536, "1.5 KB"),  # 1.5 KB
        ]

        for num_bytes, expected in test_cases:
            with self.subTest(bytes=num_bytes):
                result = format_bytes(num_bytes)
                self.assertEqual(result, expected)

    def test_append_results_new_file(self):
        """Test append_results with new file."""
        data = {"metric1": 1.0, "metric2": "test", "metric3": 42}
        append_results(self.temp_csv, data)

        # check that file was created
        self.assertTrue(self.temp_csv.exists())

        # check contents
        df = pd.read_csv(self.temp_csv)
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]["metric1"], 1.0)
        self.assertEqual(df.iloc[0]["metric2"], "test")
        self.assertEqual(df.iloc[0]["metric3"], 42)

    def test_append_results_existing_file(self):
        """Test append_results with existing file."""
        # create initial file
        data1 = {"metric1": 1.0, "metric2": "test1"}
        append_results(self.temp_csv, data1)

        # append second row
        data2 = {"metric1": 2.0, "metric2": "test2"}
        append_results(self.temp_csv, data2)

        # check contents
        df = pd.read_csv(self.temp_csv)
        self.assertEqual(len(df), 2)
        self.assertEqual(df.iloc[1]["metric1"], 2.0)
        self.assertEqual(df.iloc[1]["metric2"], "test2")

    def test_load_results_existing_file(self):
        """Test load_results with existing file."""
        data = {"metric1": 1.0, "metric2": "test"}
        append_results(self.temp_csv, data)

        df = load_results(str(self.temp_csv))
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 1)

    def test_load_results_nonexistent_file(self):
        """Test load_results with nonexistent file."""
        df = load_results("nonexistent_file.csv")
        self.assertIsNone(df)


if __name__ == "__main__":
    unittest.main()
