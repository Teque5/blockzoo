"""Tests for blockzoo.scaffold module."""

import unittest

import torch

from blockzoo.scaffold import IdentityBlock, ScaffoldNet


class TestScaffoldNet(unittest.TestCase):
    """Test cases for ScaffoldNet class."""

    def setUp(self):
        """Set up test fixtures."""
        self.block_cls = IdentityBlock
        self.input_tensor = torch.randn(2, 3, 32, 32)

    def test_scaffold_creation_all_positions(self):
        """Test ScaffoldNet creation with all positions."""
        for position in ["early", "mid", "late"]:
            with self.subTest(position=position):
                model = ScaffoldNet(self.block_cls, position=position)
                self.assertEqual(model.position, position)
                self.assertEqual(model.num_blocks, 3)  # default
                self.assertEqual(model.base_channels, 64)  # default

    def test_scaffold_forward_pass(self):
        """Test forward pass through ScaffoldNet."""
        for position in ["early", "mid", "late"]:
            with self.subTest(position=position):
                model = ScaffoldNet(self.block_cls, position=position, out_dim=10)
                output = model(self.input_tensor)

                # Check output shape
                self.assertEqual(output.shape, (2, 10))

                # Check that output is finite
                self.assertTrue(torch.isfinite(output).all())

    def test_scaffold_invalid_position(self):
        """Test ScaffoldNet with invalid position."""
        with self.assertRaises(ValueError):
            ScaffoldNet(self.block_cls, position="invalid")

    def test_scaffold_custom_parameters(self):
        """Test ScaffoldNet with custom parameters."""
        model = ScaffoldNet(self.block_cls, position="mid", num_blocks=2, base_channels=32, out_dim=100)

        self.assertEqual(model.num_blocks, 2)
        self.assertEqual(model.base_channels, 32)
        self.assertEqual(model.out_dim, 100)

        # Test forward pass with custom parameters
        output = model(self.input_tensor)
        self.assertEqual(output.shape, (2, 100))

    def test_get_stage_info(self):
        """Test get_stage_info method."""
        model = ScaffoldNet(self.block_cls, position="early", num_blocks=4)
        info = model.get_stage_info()

        expected_keys = {"position", "num_blocks", "base_channels", "out_dim", "block_class", "head_channels", "active_stage"}
        self.assertEqual(set(info.keys()), expected_keys)
        self.assertEqual(info["position"], "early")
        self.assertEqual(info["num_blocks"], 4)
        self.assertEqual(info["active_stage"], "stage1")


class TestIdentityBlock(unittest.TestCase):
    """Test cases for IdentityBlock."""

    def test_identity_block_forward(self):
        """Test IdentityBlock forward pass."""
        block = IdentityBlock(64, 64)
        x = torch.randn(2, 64, 16, 16)
        output = block(x)

        self.assertEqual(output.shape, (2, 64, 16, 16))
        self.assertTrue(torch.isfinite(output).all())

    def test_identity_block_stride(self):
        """Test IdentityBlock with stride."""
        block = IdentityBlock(64, 128, stride=2)
        x = torch.randn(2, 64, 16, 16)
        output = block(x)

        # Output should have different channels and spatial size
        self.assertEqual(output.shape, (2, 128, 8, 8))


if __name__ == "__main__":
    unittest.main()
