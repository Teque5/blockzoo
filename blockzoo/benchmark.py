"""Benchmarking utilities for BlockZoo framework.

This module provides functions to benchmark neural network models, measuring
runtime performance including latency and throughput.
"""

import argparse
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch import nn

from .scaffold import ScaffoldNet
from .utils import safe_import


def benchmark_model(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
    device: str = "cpu",
    batch_size: int = 1,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
) -> Dict[str, Any]:
    """
    Benchmark a PyTorch model for runtime performance.

    Parameters
    ----------
    model : torch.nn.Module
        The model to benchmark.
    input_shape : tuple of int, optional
        Input tensor shape as (batch_size, channels, height, width).
        Default is (1, 3, 32, 32).
    device : str, optional
        Device to run benchmark on ('cpu' or 'cuda'). Default is 'cpu'.
    batch_size : int, optional
        Batch size for benchmarking. Default is 1.
    warmup_runs : int, optional
        Number of warmup runs before benchmarking. Default is 10.
    benchmark_runs : int, optional
        Number of runs to measure for benchmarking. Default is 100.

    Returns
    -------
    dict
        Dictionary containing benchmark metrics:
        - latency_ms: Mean latency in milliseconds
        - latency_std: Standard deviation of latency in milliseconds
        - throughput: Throughput in images per second
        - device: Device used for benchmarking
        - batch_size: Batch size used

    Examples
    --------
    >>> from blockzoo.scaffold import IdentityBlock, ScaffoldNet
    >>> model = ScaffoldNet(IdentityBlock, position='mid')
    >>> results = benchmark_model(model, device='cpu')
    >>> print(f"Latency: {results['latency_ms']:.2f} ms")
    """
    model = model.to(device)
    model.eval()

    # Adjust input shape for batch size
    actual_input_shape = (batch_size, input_shape[1], input_shape[2], input_shape[3])

    # Create dummy input
    dummy_input = torch.randn(actual_input_shape).to(device)

    # Warmup runs
    print(f"[BlockZoo] Warming up with {warmup_runs} runs...")
    with torch.no_grad():
        for _ in range(warmup_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            _ = model(dummy_input)
            if device == "cuda":
                torch.cuda.synchronize()

    # Benchmark runs
    print(f"[BlockZoo] Benchmarking with {benchmark_runs} runs...")
    latencies = []

    with torch.no_grad():
        for _ in range(benchmark_runs):
            # Start timing
            if device == "cuda":
                torch.cuda.synchronize()
            start_time = time.perf_counter()

            # Forward pass
            _ = model(dummy_input)

            # End timing
            if device == "cuda":
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            # Record latency in milliseconds
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

    # Calculate statistics
    latencies = np.array(latencies)
    mean_latency = np.mean(latencies)
    std_latency = np.std(latencies)

    # Calculate throughput (images per second)
    throughput = batch_size / (mean_latency / 1000)  # Convert ms to seconds

    return {
        "latency_ms": float(mean_latency),
        "latency_std": float(std_latency),
        "throughput": float(throughput),
        "device": device,
        "batch_size": batch_size,
        "warmup_runs": warmup_runs,
        "benchmark_runs": benchmark_runs,
        "input_shape": actual_input_shape,
    }


def print_benchmark_results(results: Dict[str, Any], model_name: str = "Model") -> None:
    """
    Print benchmarking results in a formatted way.

    Parameters
    ----------
    results : dict
        Benchmark results from benchmark_model().
    model_name : str, optional
        Name of the model for display. Default is 'Model'.
    """
    print(f"\n[BlockZoo] Benchmark results for {model_name}:")
    print(f"  Device:                 {results['device']}")
    print(f"  Batch size:            {results['batch_size']}")
    print(f"  Input shape:           {results['input_shape']}")
    print(f"  Mean latency:          {results['latency_ms']:.3f} ± {results['latency_std']:.3f} ms")
    print(f"  Throughput:            {results['throughput']:.2f} images/second")
    print(f"  Benchmark runs:        {results['benchmark_runs']}")


def benchmark_block_in_scaffold(
    block_qualified_name: str,
    position: str = "mid",
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
    device: str = "cpu",
    batch_size: int = 1,
    num_blocks: int = 3,
    warmup_runs: int = 10,
    benchmark_runs: int = 100,
) -> Dict[str, Any]:
    """
    Benchmark a block wrapped in ScaffoldNet.

    Parameters
    ----------
    block_qualified_name : str
        Fully qualified name of the block class to import.
    position : str, optional
        Position to place the block ('early', 'mid', 'late'). Default is 'mid'.
    input_shape : tuple of int, optional
        Input tensor shape. Default is (1, 3, 32, 32).
    device : str, optional
        Device for benchmarking. Default is 'cpu'.
    batch_size : int, optional
        Batch size for benchmarking. Default is 1.
    num_blocks : int, optional
        Number of blocks to use in the scaffold. Default is 3.
    warmup_runs : int, optional
        Number of warmup runs. Default is 10.
    benchmark_runs : int, optional
        Number of benchmark runs. Default is 100.

    Returns
    -------
    dict
        Benchmark results with additional metadata.

    Raises
    ------
    ImportError
        If the block class cannot be imported.
    """
    # Import the block class
    block_cls = safe_import(block_qualified_name)

    # Create scaffolded model
    model = ScaffoldNet(block_cls=block_cls, position=position, num_blocks=num_blocks, base_channels=64, out_dim=10)

    # Benchmark the model
    results = benchmark_model(
        model=model, input_shape=input_shape, device=device, batch_size=batch_size, warmup_runs=warmup_runs, benchmark_runs=benchmark_runs
    )

    # Add metadata
    results.update({"block_class": block_qualified_name, "position": position, "num_blocks": num_blocks})

    return results


def benchmark_multiple_batch_sizes(
    model: nn.Module,
    batch_sizes: List[int],
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32),
    device: str = "cpu",
    warmup_runs: int = 5,
    benchmark_runs: int = 50,
) -> Dict[int, Dict[str, Any]]:
    """
    Benchmark a model across multiple batch sizes.

    Parameters
    ----------
    model : torch.nn.Module
        The model to benchmark.
    batch_sizes : list of int
        List of batch sizes to test.
    input_shape : tuple of int, optional
        Base input shape (batch_size will be overridden). Default is (1, 3, 32, 32).
    device : str, optional
        Device to run benchmark on. Default is 'cpu'.
    warmup_runs : int, optional
        Number of warmup runs per batch size. Default is 5.
    benchmark_runs : int, optional
        Number of benchmark runs per batch size. Default is 50.

    Returns
    -------
    dict
        Dictionary mapping batch sizes to their benchmark results.
    """
    results = {}

    for batch_size in batch_sizes:
        print(f"\n[BlockZoo] Benchmarking batch size {batch_size}...")

        try:
            batch_results = benchmark_model(
                model=model, input_shape=input_shape, device=device, batch_size=batch_size, warmup_runs=warmup_runs, benchmark_runs=benchmark_runs
            )
            results[batch_size] = batch_results

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"[BlockZoo] Skipping batch size {batch_size}: Out of memory")
                if device == "cuda":
                    torch.cuda.empty_cache()
                break
            else:
                raise e

    return results


def main() -> None:
    """CLI entrypoint for blockzoo-benchmark command."""
    parser = argparse.ArgumentParser(
        description="Benchmark a convolutional block wrapped in ScaffoldNet", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("block", help="Fully qualified name of the block class (e.g., 'timm.models.resnet.BasicBlock')")
    parser.add_argument("--position", choices=["early", "mid", "late"], default="mid", help="Position to place the block in scaffold")
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=4,
        default=[1, 3, 32, 32],
        metavar=("B", "C", "H", "W"),
        help="Input tensor shape (batch_size, channels, height, width)",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Device to run benchmark on")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for benchmarking")
    parser.add_argument("--num-blocks", type=int, default=3, help="Number of blocks in the scaffold stage")
    parser.add_argument("--warmup-runs", type=int, default=10, help="Number of warmup runs")
    parser.add_argument("--benchmark-runs", type=int, default=100, help="Number of benchmark runs")
    parser.add_argument("--multi-batch", nargs="+", type=int, help="Test multiple batch sizes (e.g., --multi-batch 1 2 4 8)")
    parser.add_argument("--output", help="Optional CSV file to append results to")

    args = parser.parse_args()

    # Check CUDA availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[BlockZoo] Warning: CUDA requested but not available, falling back to CPU")
        args.device = "cpu"

    try:
        if args.multi_batch:
            # Multi-batch benchmark
            block_cls = safe_import(args.block)
            model = ScaffoldNet(block_cls=block_cls, position=args.position, num_blocks=args.num_blocks, base_channels=64, out_dim=10)

            results = benchmark_multiple_batch_sizes(
                model=model,
                batch_sizes=args.multi_batch,
                input_shape=tuple(args.input_shape),
                device=args.device,
                warmup_runs=args.warmup_runs,
                benchmark_runs=args.benchmark_runs,
            )

            # Print results for each batch size
            for batch_size, batch_results in results.items():
                model_name = f"{args.block} (position={args.position}, batch={batch_size})"
                print_benchmark_results(batch_results, model_name)

                # Optionally save each result
                if args.output:
                    from .utils import append_results

                    batch_results.update({"block_class": args.block, "position": args.position, "num_blocks": args.num_blocks})
                    append_results(args.output, batch_results)

        else:
            # Single benchmark
            results = benchmark_block_in_scaffold(
                block_qualified_name=args.block,
                position=args.position,
                input_shape=tuple(args.input_shape),
                device=args.device,
                batch_size=args.batch_size,
                num_blocks=args.num_blocks,
                warmup_runs=args.warmup_runs,
                benchmark_runs=args.benchmark_runs,
            )

            # Print results
            model_name = f"{args.block} (position={args.position})"
            print_benchmark_results(results, model_name)

            # Optionally save to CSV
            if args.output:
                from .utils import append_results

                append_results(args.output, results)
                print(f"\n[BlockZoo] Results appended to {args.output}")

    except Exception as e:
        print(f"[BlockZoo] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
