"""ScaffoldNet: fixed stem → stageA → stageB → stageC → head.

This scaffold intentionally keeps stem and head identical across runs.
Blocks are injected into stages according to `position`.
"""

from typing import Type

import torch
from torch import nn


class ScaffoldNet(nn.Module):
    """
    Scaffold network that places a provided block class into one of three
    canonical stages (early, mid, late).

    Parameters
    ----------
    block_cls : type
        A class implementing a block with signature
        `block_cls(in_channels, out_channels, stride=1)`.
    position : {'early', 'mid', 'late'}
        Where to place the provided block (Stage A, B, or C).
    num_blocks : int, optional
        Number of repeated blocks in the chosen stage (default: 3).
    base_channels : int, optional
        Number of channels for the initial stem output (default: 64).
    out_dim : int, optional
        Number of output classes for the classification head (default: 10).

    Notes
    -----
    - Stages not chosen for replacement are set to `nn.Identity()` to isolate
      the block under test.
    - The scaffold uses simple downsampling schedule: StageB downsamples by 2,
      StageC downsamples by 2 from StageB.

    Examples
    --------
    >>> from blockzoo.utils import safe_import
    >>> BasicBlock = safe_import('timm.models.resnet.BasicBlock')
    >>> model = ScaffoldNet(BasicBlock, position='mid', num_blocks=2)
    >>> x = torch.randn(1, 3, 32, 32)
    >>> y = model(x)
    >>> print(y.shape)
    torch.Size([1, 10])
    """

    def __init__(
        self,
        block_cls: Type[nn.Module],
        position: str = "mid",
        num_blocks: int = 3,
        base_channels: int = 64,
        out_dim: int = 10,
    ):
        super().__init__()

        # validate position
        valid_positions = {"early", "mid", "late"}
        if position not in valid_positions:
            raise ValueError(f"Unsupported position: {position!r}. Must be one of {valid_positions}")

        self.position = position
        self.num_blocks = num_blocks
        self.base_channels = base_channels
        self.out_dim = out_dim
        self.block_cls = block_cls

        # fixed stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        def make_stage(ch_in: int, ch_out: int, stride: int) -> nn.Sequential:
            """Create a stage with repeated blocks."""
            layers = [block_cls(ch_in, ch_out, stride=stride)]
            for _ in range(num_blocks - 1):
                layers.append(block_cls(ch_out, ch_out, stride=1))
            return nn.Sequential(*layers)

        c = base_channels
        stage_a = make_stage(c, c, stride=1)  # early-stage (high-res)
        stage_b = make_stage(c, 2 * c, stride=2)  # mid-stage (downsample)
        stage_c = make_stage(2 * c, 4 * c, stride=2)  # late-stage (low-res)

        # create default stages using BasicBlock for non-target positions
        def make_basic_stage(ch_in: int, ch_out: int, stride: int) -> nn.Sequential:
            """Create a stage with repeated BasicBlocks."""
            layers = [BasicBlock(ch_in, ch_out, stride=stride)]
            for _ in range(num_blocks - 1):
                layers.append(BasicBlock(ch_out, ch_out, stride=1))
            return nn.Sequential(*layers)

        # set stages based on position with proper channel handling
        if position == "early":
            self.stage1 = stage_a  # target blocks: 64 -> 64
            self.stage2 = make_basic_stage(c, 2 * c, stride=2)  # basicBlock: 64 -> 128
            self.stage3 = make_basic_stage(2 * c, 4 * c, stride=2)  # basicBlock: 128 -> 256
            self._head_channels = 4 * c
        elif position == "mid":
            self.stage1 = make_basic_stage(c, c, stride=1)  # basicBlock: 64 -> 64
            self.stage2 = stage_b  # target blocks: 64 -> 128
            self.stage3 = make_basic_stage(2 * c, 4 * c, stride=2)  # basicBlock: 128 -> 256
            self._head_channels = 4 * c
        elif position == "late":
            self.stage1 = make_basic_stage(c, c, stride=1)  # basicBlock: 64 -> 64
            self.stage2 = make_basic_stage(c, 2 * c, stride=2)  # basicBlock: 64 -> 128
            # adjust stage C to start from stage B output
            stage_c = make_stage(2 * c, 4 * c, stride=2)  # target blocks: 128 -> 256
            self.stage3 = stage_c
            self._head_channels = 4 * c

        # head: adapt to the channels produced by the active stage
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self._head_channels, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through scaffold.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, 3, height, width).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, out_dim).
        """
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)

    def get_stage_info(self) -> dict:
        """
        Get information about the scaffold configuration.

        Returns
        -------
        dict
            Dictionary containing scaffold configuration details.
        """
        return {
            "position": self.position,
            "num_blocks": self.num_blocks,
            "base_channels": self.base_channels,
            "out_dim": self.out_dim,
            "block_class": self.block_cls.__name__,
            "head_channels": self._head_channels,
            "active_stage": f'stage{["early", "mid", "late"].index(self.position) + 1}',
        }


class BasicBlock(nn.Module):
    """
    Simple basic block for testing purposes and as stand-in for non-target stages.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    stride : int, optional
        Stride for convolution (default: 1).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))
