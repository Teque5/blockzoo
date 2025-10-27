"""Tests for blockzoo.benchmark module."""

import unittest

from blockzoo.benchmark import benchmark_block_in_scaffold, benchmark_model
from blockzoo.scaffold import ScaffoldNet
from blockzoo.wrappers import ResNetBasicBlockWrapper


class TestBenchmark(unittest.TestCase):
    """Test cases for benchmark module."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = ScaffoldNet(ResNetBasicBlockWrapper, position="mid")
        self.input_shape = (1, 3, 32, 32)

    def test_benchmark_model(self):
        """Test benchmark_model function."""
        # use minimal runs for testing
        results = benchmark_model(self.model, self.input_shape, device="cpu", warmup_runs=2, benchmark_runs=5)

        # check required keys
        required_keys = {"latency_ms", "latency_std", "throughput", "device", "batch_size", "warmup_runs", "benchmark_runs", "input_shape"}
        self.assertEqual(set(results.keys()), required_keys)

        # check that values are reasonable
        self.assertGreater(results["latency_ms"], 0)
        self.assertGreaterEqual(results["latency_std"], 0)
        self.assertGreater(results["throughput"], 0)
        self.assertEqual(results["device"], "cpu")

    def test_benchmark_block_in_scaffold(self):
        """Test benchmark_block_in_scaffold function."""
        results = benchmark_block_in_scaffold("ResNetBasicBlock", position="mid", warmup_runs=2, benchmark_runs=5)

        # check that additional metadata is included
        self.assertEqual(results["block_class"], "ResNetBasicBlock")
        self.assertEqual(results["position"], "mid")


if __name__ == "__main__":
    unittest.main()
