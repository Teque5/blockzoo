"""Model profiling utilities for BlockZoo framework.

This module provides functions to profile neural network models, measuring
parameters, FLOPs, and memory usage using various profiling libraries.
"""

import argparse
import sys
from typing import Any, Dict, Tuple

import torch
from timm.utils.model import reparameterize_model
from torch import nn
from torchinfo import summary

from .scaffold import ScaffoldNet
from .utils import append_results, format_bytes
from .wrappers import ResNetBasicBlockWrapper, get_block_class


def quick_profile(block_class_name: str, position: str = "mid") -> dict:
    """
    Quick profiling of a block in ScaffoldNet.

    Parameters
    ----------
    block_class_name : str
        Fully qualified name of the block class.
    position : str, optional
        Position to place the block ('early', 'mid', 'late'). Default is 'mid'.

    Returns
    -------
    dict
        Profiling results.

    Examples
    --------
    >>> results = quick_profile('ResNetBasicBlock', 'mid')
    >>> results['params_total']
    4505866
    """
    return profile_block_in_scaffold(block_name=block_class_name, position=position)


def get_model_profile(model: nn.Module, input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32), device: str = "cpu") -> Dict[str, Any]:
    """
    Get comprehensive profiling information for a PyTorch model.

    Parameters
    ----------
    model : torch.nn.Module
        The model to profile.
    input_shape : tuple of int, optional
        Input tensor shape as (batch_size, channels, height, width).
        Default is (1, 3, 32, 32).
    device : str, optional
        Device to run profiling on ('cpu' or 'cuda'). Default is 'cpu'.

    Returns
    -------
    dict
        Dictionary containing profiling metrics:
        - params_total: Total number of parameters
        - params_trainable: Number of trainable parameters
        - flops: Number of FLOPs (floating point operations)
        - memory_mb: Estimated memory usage in MB

    Examples
    --------
    >>> model = ScaffoldNet(ResNetBasicBlockWrapper, position='mid')
    >>> profile = get_model_profile(model)
    >>> profile['params_total']
    4505866
    """
    # reparameterize model if supported
    model = reparameterize_model(model)
    model = model.to(device)
    model.eval()

    profile = {"params_total": 0, "params_trainable": 0, "flops": 0, "memory_mb": 0.0}

    # count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    profile["params_total"] = total_params
    profile["params_trainable"] = trainable_params

    # create dummy input
    dummy_input = torch.randn(input_shape).to(device)

    # method 1: Use torchinfo (most comprehensive)
    with torch.no_grad():
        model_summary = summary(model, input_size=input_shape, device=device, verbose=0)
        profile["memory_mb"] = model_summary.total_param_bytes / (1024**2)
        # rough estimate: 4 bytes per float32 parameter
        profile["memory_mb"] = (model_summary.total_params * 4) / (1024**2)
        # to quote NVIDIA: "Each multiply-add comprises two operations, thus one would multiply the throughput in the table by 2 to get FLOP counts per clock."
        profile["flops"] = model_summary.total_mult_adds * 2

    return profile


def print_profile(profile: Dict[str, Any], model_name: str = "Model") -> None:
    """
    Print model profiling results in a formatted way.

    Parameters
    ----------
    profile : dict
        Profiling results from get_model_profile().
    model_name : str, optional
        Name of the model for display. Default is 'Model'.
    """
    print(f"\n[BlockZoo] Profile for {model_name}:")
    print(f"  Parameters (total):     {profile['params_total']:,}")
    print(f"  Parameters (trainable): {profile['params_trainable']:,}")
    print(f"  FLOPs:                  {profile['flops']:,}")
    print(f"  Memory estimate:        {format_bytes(int(profile['memory_mb'] * 1024 * 1024))}")
    print(f"  Memory (MB):           {profile['memory_mb']:.2f}")


def profile_block_in_scaffold(
    block_name: str, position: str = "mid", input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32), device: str = "cpu", num_blocks: int = 3
) -> Dict[str, Any]:
    """
    Profile a block wrapped in ScaffoldNet.

    Parameters
    ----------
    block_name : str
        Name of the block in the block registry (e.g., 'InvertedResidual').
    position : str, optional
        Position to place the block ('early', 'mid', 'late'). Default is 'mid'.
    input_shape : tuple of int, optional
        Input tensor shape. Default is (1, 3, 32, 32).
    device : str, optional
        Device for profiling. Default is 'cpu'.
    num_blocks : int, optional
        Number of blocks to use in the scaffold. Default is 3.

    Returns
    -------
    dict
        Profiling results with additional metadata.

    Raises
    ------
    ImportError
        If the block class cannot be imported.
    """
    # get the block class from registry
    block_cls = get_block_class(block_name)

    # create scaffolded model
    model = ScaffoldNet(block_cls=block_cls, position=position, num_blocks=num_blocks, base_channels=64, out_dim=10)

    # get profile
    profile = get_model_profile(model, input_shape, device)

    # add metadata
    profile.update({"block_class": block_name, "position": position, "num_blocks": num_blocks, "input_shape": input_shape, "device": device})

    return profile


def main() -> None:
    """CLI entrypoint for blockzoo-profile command."""
    parser = argparse.ArgumentParser(description="Profile a convolutional block wrapped in ScaffoldNet", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("block", help="Block Name (e.g., 'ResNetBasicBlock')")
    parser.add_argument("--position", choices=["early", "mid", "late"], default="mid", help="Position to place the block in scaffold")
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=4,
        default=[1, 3, 32, 32],
        metavar=("B", "C", "H", "W"),
        help="Input tensor shape (batch_size, channels, height, width)",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device to run profiling on")
    parser.add_argument("--num-blocks", type=int, default=3, help="Number of blocks in the scaffold stage")
    parser.add_argument("--output", help="Optional CSV file to append results to")

    args = parser.parse_args()

    # check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[BlockZoo] Warning: CUDA requested but not available, falling back to CPU")
        args.device = "cpu"

    try:
        # profile the block
        profile = profile_block_in_scaffold(
            block_name=args.block, position=args.position, input_shape=tuple(args.input_shape), device=args.device, num_blocks=args.num_blocks
        )

        # print results
        model_name = f"{args.block} (position={args.position})"
        print_profile(profile, model_name)

        # optionally save to CSV
        if args.output:

            append_results(args.output, profile)
            print(f"\n[BlockZoo] Results appended to {args.output}")

    except Exception as e:
        print(f"[BlockZoo] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
