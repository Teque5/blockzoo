"""
Block wrappers for external block architectures.

This module provides a unified interface for blocks from different libraries
(like timm) that may have different constructor signatures than what BlockZoo expects.
The lookup table maps block names to wrapper functions that return properly configured blocks.
"""

from functools import partial
from typing import Any, Callable, Dict

from torch import nn

# import timm blocks
from timm.models._efficientnet_blocks import InvertedResidual, UniversalInvertedResidual, EdgeResidual
from timm.models.resnet import BasicBlock as TimmBasicBlock, Bottleneck

# import blockzoo blocks
from .scaffold import BasicBlock as BlockZooBasicBlock


def create_downsample_if_needed(in_channels: int, out_channels: int, stride: int) -> nn.Module:
    """Create downsample layer for shortcut connection if channels or stride don't match."""
    if stride != 1 or in_channels != out_channels:
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels))
    return None


# simple custom blocks for fallback when timm blocks aren't available
class SimpleResidualBlock(nn.Module):
    """Simple residual block similar to ResNet BasicBlock."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False), nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)


# wrapper classes that adapt timm blocks to BlockZoo interface
class InvertedResidualWrapper(nn.Module):
    """Wrapper for timm InvertedResidual with BlockZoo interface."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = InvertedResidual(
            in_chs=in_channels,
            out_chs=out_channels,
            stride=stride,
            exp_ratio=6.0  # standard MobileNet expansion ratio
        )

    def forward(self, x):
        return self.block(x)


class UniversalInvertedResidualWrapper(nn.Module):
    """Wrapper for timm UniversalInvertedResidual with BlockZoo interface."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = UniversalInvertedResidual(
            in_chs=in_channels,
            out_chs=out_channels,
            stride=stride,
            exp_ratio=4.0  # standard expansion ratio for UIB
        )

    def forward(self, x):
        return self.block(x)


class EdgeResidualWrapper(nn.Module):
    """Wrapper for timm EdgeResidual with BlockZoo interface."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = EdgeResidual(
            in_chs=in_channels,
            out_chs=out_channels,
            stride=stride,
            exp_ratio=6.0  # standard expansion ratio
        )

    def forward(self, x):
        return self.block(x)


class ResNetBasicBlockWrapper(nn.Module):
    """Wrapper for timm ResNet BasicBlock with BlockZoo interface."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # create downsample if needed
        downsample = create_downsample_if_needed(in_channels, out_channels, stride)

        self.block = TimmBasicBlock(
            inplanes=in_channels,
            planes=out_channels,
            stride=stride,
            downsample=downsample
        )

    def forward(self, x):
        return self.block(x)


class ResNetBottleneckWrapper(nn.Module):
    """Wrapper for timm ResNet Bottleneck with BlockZoo interface."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        # bottleneck expects planes (where planes * 4 = out_channels)
        planes = out_channels // 4
        if out_channels % 4 != 0:
            planes = out_channels // 4 + 1
            out_channels = planes * 4  # adjust to valid bottleneck output

        # create downsample if needed
        downsample = create_downsample_if_needed(in_channels, out_channels, stride)

        self.block = Bottleneck(
            inplanes=in_channels,
            planes=planes,
            stride=stride,
            downsample=downsample
        )

    def forward(self, x):
        return self.block(x)




# block lookup table - maps block names to their wrapper classes
BLOCK_REGISTRY: Dict[str, Callable[[int, int, int], nn.Module]] = {
    # timm EfficientNet blocks
    "InvertedResidual": InvertedResidualWrapper,
    "UniversalInvertedResidual": UniversalInvertedResidualWrapper,
    "EdgeResidual": EdgeResidualWrapper,
    # timm ResNet blocks
    "ResNetBasicBlock": ResNetBasicBlockWrapper,
    "ResNetBottleneck": ResNetBottleneckWrapper,
    # custom simple blocks (fallbacks)
    "SimpleResidualBlock": SimpleResidualBlock,

    # blockzoo basic block (for tests and simple cases)
    "BasicBlock": BlockZooBasicBlock,
}


def get_block_class(block_name: str) -> Callable[[int, int, int], nn.Module]:
    """
    Get a block creator function by name.

    Args:
        block_name: Name of the block (e.g., 'InvertedResidual', 'UniversalInvertedResidual')

    Returns:
        Function that takes (in_channels, out_channels, stride) and returns a block

    Raises:
        KeyError: If block_name is not found in the registry
    """
    if block_name not in BLOCK_REGISTRY:
        available_blocks = list(BLOCK_REGISTRY.keys())
        raise KeyError(f"Block '{block_name}' not found. Available blocks: {available_blocks}")

    return BLOCK_REGISTRY[block_name]


def list_available_blocks() -> list[str]:
    """List all available block names."""
    return list(BLOCK_REGISTRY.keys())


def create_block(block_name: str, in_channels: int, out_channels: int, stride: int = 1, **kwargs) -> nn.Module:
    """
    Create a block by name with standardized (in_channels, out_channels, stride) interface.

    Args:
        block_name: Name of the block
        in_channels: Input channels
        out_channels: Output channels
        stride: Stride for the block
        **kwargs: Additional arguments passed to the block constructor

    Returns:
        Configured block module
    """
    block_creator = get_block_class(block_name)
    return block_creator(in_channels, out_channels, stride, **kwargs)
