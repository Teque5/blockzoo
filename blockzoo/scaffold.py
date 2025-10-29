"""ScaffoldNet: fixed stem → stageA → stageB → stageC → head.

This scaffold intentionally keeps stem and head identical across runs.
Blocks are injected into stages according to `position`.
"""

from typing import Type

import torch
from torch import nn

from .wrappers import ResNetBasicBlockWrapper


class ScaffoldNet(nn.Module):
    """
    Scaffold network that places a provided block class into one of three
    canonical stages (early, mid, late).

    Parameters
    ----------
    block_cls : type
        A class implementing a block with signature with kwargs (in_channels, out_channels, stride).
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
    >>> model = ScaffoldNet(ResNetBasicBlockWrapper, position='mid')
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

        self.position = position
        self.num_blocks = num_blocks
        self.channels = [base_channels] + [base_channels * 2**sdx for sdx in range(3)]
        self.out_dim = out_dim
        self.block_cls = block_cls

        # fixed stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        def _make_stage(in_channels: int, out_channels: int, stride: int, stage_position: str) -> nn.Sequential:
            """Create a stage with repeated blocks."""
            layers = [block_cls(in_channels, out_channels, stride=stride, position=stage_position)]
            for _ in range(num_blocks - 1):
                layers.append(block_cls(out_channels, out_channels, stride=1, position=stage_position))
            return nn.Sequential(*layers)

        # set stages based on position with proper channel handling
        if position == "early":
            # target_block in stage 1 (local features)
            self.stage1 = self._make_stage(block_cls, self.channels[0], self.channels[1], 1, "early")
            self.stage2 = self._make_stage(ResNetBasicBlockWrapper, self.channels[1], self.channels[2], 2, "mid")
            self.stage3 = self._make_stage(ResNetBasicBlockWrapper, self.channels[2], self.channels[3], 2, "late")
        elif position == "mid":
            # target_block in stage 2 (mid-level features)
            self.stage1 = self._make_stage(ResNetBasicBlockWrapper, self.channels[0], self.channels[1], 1, "early")
            self.stage2 = self._make_stage(block_cls, self.channels[1], self.channels[2], 2, "mid")
            self.stage3 = self._make_stage(ResNetBasicBlockWrapper, self.channels[2], self.channels[3], 2, "late")
        else:
            # target_block in stage 3 (global features)
            self.stage1 = self._make_stage(ResNetBasicBlockWrapper, self.channels[0], self.channels[1], 1, "early")
            self.stage2 = self._make_stage(ResNetBasicBlockWrapper, self.channels[1], self.channels[2], 2, "mid")
            self.stage3 = self._make_stage(block_cls, self.channels[2], self.channels[3], 2, "late")

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.channels[3], out_dim),
        )

    def _make_stage(self, block_cls: Type[nn.Module], in_channels: int, out_channels: int, stride: int, stage_position: str) -> nn.Sequential:
        """Create a stage with repeated blocks."""
        layers = [block_cls(in_channels, out_channels, stride=stride, position=stage_position)]
        for _ in range(self.num_blocks - 1):
            layers.append(block_cls(out_channels, out_channels, stride=1, position=stage_position))
        return nn.Sequential(*layers)

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
            "base_channels": self.channels[0],
            "out_dim": self.out_dim,
            "block_class": self.block_cls.__name__,
            "head_channels": self.channels[-1],
            "active_stage": f'stage{["early", "mid", "late"].index(self.position) + 1}',
        }
