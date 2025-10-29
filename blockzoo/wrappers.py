"""Unified interface for blocks from different libraries. See lookup table BLOCK_REGISTRY."""

from functools import partial
from typing import Any, Callable, Dict

from timm.models._efficientnet_blocks import EdgeResidual, InvertedResidual, SqueezeExcite, UniversalInvertedResidual
from timm.models.fastvit import FastVitStage, MobileOneBlock, ReparamLargeKernelConv
from timm.models.ghostnet import GhostBottleneckV3
from timm.models.resnet import BasicBlock as ResNetBasicBlock
from timm.models.resnet import Bottleneck as ResNetBottleneck
from timm.models.resnet import downsample_avg
from timm.models.vitamin import MbConvLNBlock
from torch import nn


class EdgeResidualWrapper(EdgeResidual):
    """
    EdgeResidual from EfficientNet-EdgeTPU and EfficientNetV2, aka FusedMBConv
    Very early layers (16ch) use exp_ratio of 1.0, so full range is in [1, 6].
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int, position: str):
        exp_ratio = 6.0 if position == "late" else 4.0
        se_ratio = 0.25 if position == "late" else 0.0
        se_layer = partial(SqueezeExcite, rd_ratio=se_ratio) if se_ratio else None

        super().__init__(
            in_chs=in_channels,
            out_chs=out_channels,
            stride=stride,
            exp_ratio=exp_ratio,
            se_layer=se_layer,
        )


class InvertedResidualWrapper(InvertedResidual):
    """InvertedResidual from MobileNetV3 and EfficientNet, aka MBConv"""

    def __init__(self, in_channels: int, out_channels: int, stride: int, position: str):
        exp_ratio = 6.0 if position == "late" else 4.0
        act_layer = nn.ReLU if position == "early" else nn.Hardswish
        super().__init__(
            in_chs=in_channels,
            out_chs=out_channels,
            stride=stride,
            exp_ratio=exp_ratio,
            act_layer=act_layer,
        )


class GhostBottleneckV3Wrapper(GhostBottleneckV3):
    """
    GhostBottleneckV3 from GhostNetV3
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int, position: str):
        expansion = 5 if position == "late" else 3
        se_ratio = 0.25 if position == "late" else 0
        super().__init__(
            in_chs=in_channels,
            mid_chs=in_channels * expansion,
            out_chs=out_channels,
            stride=stride,
            se_ratio=se_ratio,
        )


class MbConvLNBlockWrapper(MbConvLNBlock):
    """MbConvLNBlock from ViTamin, similar to InvertedResidual but with LayerNorm"""

    def __init__(self, in_channels: int, out_channels: int, stride: int, position: str):
        super().__init__(
            in_chs=in_channels,
            out_chs=out_channels,
            stride=stride,
        )


class MobileOneBlockWrapper(MobileOneBlock):
    """MobileOneBlock from MobileOne architecture"""

    def __init__(self, in_channels: int, out_channels: int, stride: int, position: str):
        super().__init__(
            in_chs=in_channels,
            out_chs=out_channels,
            kernel_size=3,
            stride=stride,
        )


class ReparamLargeKernelConvWrapper(ReparamLargeKernelConv):
    """ReparamLargeKernelConv from RepLKNet"""

    def __init__(self, in_channels: int, out_channels: int, stride: int, position: str):
        super().__init__(
            in_chs=in_channels,
            out_chs=out_channels,
            kernel_size=7,
            small_kernel=3,
            stride=stride,
            group_size=1,
            act_layer=nn.GELU,
        )


class RepMixerWrapper(FastVitStage):
    """RepMixer from FastViT"""

    def __init__(self, in_channels: int, out_channels: int, stride: int, position: str):
        super().__init__(
            dim=in_channels,
            dim_out=out_channels,
            depth=1,
            token_mixer_type="repmixer",
            mlp_ratio=4.0,
            down_stride=stride,
            drop_path_rate=[0.0],
        )


class ResNetBasicBlockWrapper(ResNetBasicBlock):
    """ResNet BasicBlock from the foundational 2015 ResNet paper"""

    def __init__(self, in_channels: int, out_channels: int, stride: int, position: str):
        super().__init__(
            inplanes=in_channels,
            planes=out_channels,
            stride=stride,
            downsample=downsample_avg(in_channels, out_channels, 1, stride=stride),
        )


class ResNetBottleneckWrapper(ResNetBottleneck):
    """ResNet BottleneckBlock for deeper ResNet architectures"""

    def __init__(self, in_channels: int, out_channels: int, stride: int, position: str):
        # Note: ResNet Bottleneck uses planes where final output is planes * expansion (4),
        # so as long as our in_channels are divisible by 4, we can set planes = out_channels // 4
        super().__init__(
            inplanes=in_channels,
            planes=out_channels // 4,
            stride=stride,
            downsample=downsample_avg(in_channels, out_channels, 1, stride=stride),
        )


class UniversalInvertedBottleneckWrapper(UniversalInvertedResidual):
    """
    UniversalInvertedBottleneck from MobileNetV4, aka UIB
    Very early layers (16ch) use exp_ratio of 2.0, so full range is in [2, 6].
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int, position: str):
        exp_ratio = 6.0 if position == "late" else 4.0
        super().__init__(
            in_chs=in_channels,
            out_chs=out_channels,
            stride=stride,
            exp_ratio=exp_ratio,
        )


# block lookup table - maps block names to their wrapper classes
BLOCK_REGISTRY: Dict[str, Callable[[int, int, int, str], nn.Module]] = {
    "EdgeResidual": EdgeResidualWrapper,
    "GhostBottleneckV3": GhostBottleneckV3Wrapper,
    "InvertedResidual": InvertedResidualWrapper,
    "MBConvLNBlock": MbConvLNBlockWrapper,
    "MobileOneBlock": MobileOneBlockWrapper,
    "ReparamLargeKernelConv": ReparamLargeKernelConvWrapper,
    "RepMixer": RepMixerWrapper,
    "ResNetBasicBlock": ResNetBasicBlockWrapper,
    "ResNetBottleneck": ResNetBottleneckWrapper,
    "UniversalInvertedBottleneck": UniversalInvertedBottleneckWrapper,
}


def get_block_class(block_name: str) -> Callable[[int, int, int, str], nn.Module]:
    """
    Get a block creator function by name.

    Args:
        block_name: Name of the block (e.g., 'InvertedResidual', 'UniversalInvertedResidual')

    Returns:
        Function that takes (in_channels, out_channels, stride, position) and returns a block

    Raises:
        KeyError: If block_name is not found in the registry
    """
    if block_name not in BLOCK_REGISTRY:
        available_blocks = list(BLOCK_REGISTRY.keys())
        raise KeyError(f"Block '{block_name}' not found. Available blocks: {available_blocks}")

    return BLOCK_REGISTRY[block_name]
