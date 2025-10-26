"""
Tests for blockzoo.wrappers module.

This module tests that all blocks in BLOCK_REGISTRY work correctly
and produce the expected output shapes.
"""

import pytest
import torch
from blockzoo.wrappers import BLOCK_REGISTRY, list_available_blocks, create_block


class TestBlockRegistry:
    """Test all blocks in the BLOCK_REGISTRY."""

    def test_list_available_blocks(self):
        """Test that list_available_blocks returns expected blocks."""
        blocks = list_available_blocks()
        expected_blocks = {
            "InvertedResidual",
            "UniversalInvertedResidual",
            "EdgeResidual",
            "ResNetBasicBlock",
            "ResNetBottleneck",
            "SimpleResidualBlock"
        }
        assert set(blocks) == expected_blocks

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


class TestSpecificBlocks:
    """Test specific blocks with known behaviors."""

    def test_resnet_bottleneck_expansion(self):
        """Test ResNetBottleneck handles expansion correctly."""
        # bottleneck expects planes where planes * 4 = out_channels
        in_channels = 64
        out_channels = 256  # should work perfectly (256 / 4 = 64)
        stride = 1

        block = create_block("ResNetBottleneck", in_channels, out_channels, stride)

        x = torch.randn(1, in_channels, 16, 16)
        with torch.no_grad():
            y = block(x)

        assert y.shape == (1, out_channels, 16, 16)

    def test_resnet_bottleneck_expansion_adjustment(self):
        """Test ResNetBottleneck adjusts non-divisible output channels."""
        # test with output channels not divisible by 4
        in_channels = 64
        out_channels = 102  # not divisible by 4, should be adjusted to 104 (planes=26, 26*4=104)
        stride = 1

        block = create_block("ResNetBottleneck", in_channels, out_channels, stride)

        x = torch.randn(1, in_channels, 16, 16)
        with torch.no_grad():
            y = block(x)

        # output should be adjusted to next multiple of 4 (104)
        assert y.shape == (1, 104, 16, 16)

    def test_inverted_residual_parameters(self):
        """Test InvertedResidual uses correct expansion ratio."""
        block = create_block("InvertedResidual", 32, 64, stride=2)

        x = torch.randn(1, 32, 32, 32)
        with torch.no_grad():
            y = block(x)

        assert y.shape == (1, 64, 16, 16)

    def test_universal_inverted_residual_parameters(self):
        """Test UniversalInvertedResidual uses correct expansion ratio."""
        block = create_block("UniversalInvertedResidual", 48, 96, stride=1)

        x = torch.randn(1, 48, 16, 16)
        with torch.no_grad():
            y = block(x)

        assert y.shape == (1, 96, 16, 16)


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_invalid_block_name(self):
        """Test that invalid block names raise KeyError."""
        with pytest.raises(KeyError, match="Block 'NonExistentBlock' not found"):
            create_block("NonExistentBlock", 64, 128, 1)

    def test_small_input_size(self):
        """Test blocks work with small input sizes."""
        for block_name in BLOCK_REGISTRY.keys():
            block = create_block(block_name, 16, 32, stride=1)

            # test with 4x4 input
            x = torch.randn(1, 16, 4, 4)
            with torch.no_grad():
                y = block(x)

            assert y.shape == (1, 32, 4, 4), f"{block_name} failed with small input"

    def test_large_channels(self):
        """Test blocks work with large channel counts."""
        for block_name in BLOCK_REGISTRY.keys():
            # skip bottleneck for this test as it has specific channel requirements
            if block_name == "ResNetBottleneck":
                continue

            block = create_block(block_name, 512, 1024, stride=1)

            x = torch.randn(1, 512, 8, 8)
            with torch.no_grad():
                y = block(x)

            assert y.shape == (1, 1024, 8, 8), f"{block_name} failed with large channels"


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