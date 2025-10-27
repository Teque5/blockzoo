"""
Tests for blockzoo.wrappers module.

This module tests that all blocks in BLOCK_REGISTRY work correctly
and produce the expected output shapes.
"""

import pytest
import torch

from blockzoo.wrappers import BLOCK_REGISTRY, create_block, list_available_blocks


class TestBlockRegistry:
    """Test all blocks in the BLOCK_REGISTRY."""

    @pytest.mark.parametrize("block_name", list(BLOCK_REGISTRY.keys()))
    def test_block_shapes_stride_1(self, block_name):
        """Test that all blocks work with stride=1 and same channels."""
        in_channels, out_channels = 64, 64
        stride = 1

        # create block
        block = create_block(block_name, in_channels, out_channels, stride)

        # test forward pass
        x = torch.randn(2, in_channels, 32, 32)
        with torch.no_grad():
            y = block(x)

        # check output shape
        expected_shape = (2, out_channels, 32, 32)
        assert y.shape == expected_shape, f"{block_name} stride=1: expected {expected_shape}, got {y.shape}"

    @pytest.mark.parametrize("block_name", list(BLOCK_REGISTRY.keys()))
    def test_block_shapes_stride_2(self, block_name):
        """Test that all blocks work with stride=2 and different channels."""
        in_channels, out_channels = 64, 128
        stride = 2

        # create block
        block = create_block(block_name, in_channels, out_channels, stride)

        # test forward pass
        x = torch.randn(2, in_channels, 32, 32)
        with torch.no_grad():
            y = block(x)

        # check output shape (spatial dims should be halved due to stride=2)
        expected_shape = (2, out_channels, 16, 16)
        assert y.shape == expected_shape, f"{block_name} stride=2: expected {expected_shape}, got {y.shape}"

    @pytest.mark.parametrize("block_name", list(BLOCK_REGISTRY.keys()))
    def test_block_shapes_channel_change(self, block_name):
        """Test that all blocks work with channel dimension changes."""
        in_channels, out_channels = 128, 256
        stride = 1

        # create block
        block = create_block(block_name, in_channels, out_channels, stride)

        # test forward pass
        x = torch.randn(1, in_channels, 16, 16)
        with torch.no_grad():
            y = block(x)

        # check output shape
        expected_shape = (1, out_channels, 16, 16)
        assert y.shape == expected_shape, f"{block_name} channel change: expected {expected_shape}, got {y.shape}"


if __name__ == "__main__":
    # quick test to run manually
    print("Testing all blocks in BLOCK_REGISTRY...")

    for block_name in BLOCK_REGISTRY.keys():
        try:
            # test basic functionality
            block = create_block(block_name, 64, 128, stride=2)
            x = torch.randn(1, 64, 16, 16)
            y = block(x)
            print(f"✅ {block_name}: {y.shape}")
        except Exception as e:
            print(f"❌ {block_name}: {e}")

    print("Done!")
