"""Tests for blockzoo.profiler module."""

import unittest

from blockzoo.profiler import get_model_profile, profile_block_in_scaffold
from blockzoo.scaffold import BasicBlock, ScaffoldNet


class TestProfiler(unittest.TestCase):
    """Test cases for profiler module."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = ScaffoldNet(BasicBlock, position="mid")
        self.input_shape = (1, 3, 32, 32)

    def test_get_model_profile(self):
        """Test get_model_profile function."""
        profile = get_model_profile(self.model, self.input_shape)

        # check required keys
        required_keys = {"params_total", "params_trainable", "flops", "memory_mb"}
        self.assertEqual(set(profile.keys()), required_keys)

        # check that values are reasonable
        self.assertGreater(profile["params_total"], 0)
        self.assertGreaterEqual(profile["params_trainable"], 0)
        self.assertGreaterEqual(profile["flops"], 0)
        self.assertGreaterEqual(profile["memory_mb"], 0.0)

        # trainable params should not exceed total params
        self.assertLessEqual(profile["params_trainable"], profile["params_total"])

    def test_profile_block_in_scaffold(self):
        """Test profile_block_in_scaffold function."""
        profile = profile_block_in_scaffold("blockzoo.scaffold.BasicBlock", position="mid")

        # check that additional metadata is included
        self.assertEqual(profile["block_class"], "blockzoo.scaffold.BasicBlock")
        self.assertEqual(profile["position"], "mid")
        self.assertEqual(profile["num_blocks"], 3)  # default


if __name__ == "__main__":
    unittest.main()
