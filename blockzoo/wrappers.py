"""
Block wrappers for external block architectures.

This module provides a unified interface for blocks from different libraries
(like timm) that may have different constructor signatures than what BlockZoo expects.
The lookup table maps block names to wrapper functions that return properly configured blocks.
"""

from functools import partial
from typing import Any, Callable, Dict

from timm.models._efficientnet_blocks import EdgeResidual, InvertedResidual, UniversalInvertedResidual
from timm.models.resnet import BasicBlock as ResNetBasicBlock
from timm.models.resnet import Bottleneck as ResNetBottleneck
from timm.models.resnet import downsample_avg
from torch import nn


class InvertedResidualWrapper(InvertedResidual):
    """InvertedResidual from MobileNetV2 and EfficientNetV2, aka MBConv"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            in_chs=in_channels,
            out_chs=out_channels,
            stride=stride,
            exp_ratio=6.0,  # standard MobileNet expansion ratio
        )


class UniversalInvertedBottleneckWrapper(UniversalInvertedResidual):
    """UniversalInvertedBottleneck from MobileNetV4, aka UIB"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            in_chs=in_channels,
            out_chs=out_channels,
            stride=stride,
            exp_ratio=4.0,  # standard expansion ratio for UIB
        )


class EdgeResidualWrapper(EdgeResidual):
    """EdgeResidual from EfficientNet-Edge and EfficientNetV2, aka FusedMBConv"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            in_chs=in_channels,
            out_chs=out_channels,
            stride=stride,
            exp_ratio=6.0,  # standard expansion ratio
        )


class ResNetBasicBlockWrapper(ResNetBasicBlock):
    """ResNet BasicBlock from the foundational 2015 ResNet paper"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__(
            inplanes=in_channels,
            planes=out_channels,
            stride=stride,
            downsample=downsample_avg(in_channels, out_channels, 1, stride=stride),
        )


class ResNetBottleneckWrapper(ResNetBottleneck):
    """ResNet BottleneckBlock for deeper ResNet architectures from the 2015 ResNet paper"""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        # Note: ResNet Bottleneck uses planes where final output is planes * expansion (4),
        # so as long as our in_channels are divisible by 4, we can set planes = out_channels // 4
        super().__init__(
            inplanes=in_channels,
            planes=out_channels // 4,
            stride=stride,
            downsample=downsample_avg(in_channels, out_channels, 1, stride=stride),
        )


# block lookup table - maps block names to their wrapper classes
BLOCK_REGISTRY: Dict[str, Callable[[int, int, int], nn.Module]] = {
    "InvertedResidual": InvertedResidualWrapper,
    "UniversalInvertedBottleneck": UniversalInvertedBottleneckWrapper,
    "EdgeResidual": EdgeResidualWrapper,
    "ResNetBasicBlock": ResNetBasicBlockWrapper,
    "ResNetBottleneck": ResNetBottleneckWrapper,
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
