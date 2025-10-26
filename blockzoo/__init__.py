"""BlockZoo: Convolutional Block Benchmarking Framework.

BlockZoo is a framework for benchmarking and profiling convolutional building
blocks in isolation, measuring how their positional specialization (early/mid/late)
affects feature extraction capability, with FLOPs/params/memory/runtime recorded.
"""

__version__ = "0.1.0"
__description__ = "Benchmark and profile convolutional building blocks for feature extraction."

from .benchmark import benchmark_block_in_scaffold, benchmark_model
from .config import BenchmarkConfig, ExperimentConfig, get_dataset_config
from .profiler import get_model_profile, profile_block_in_scaffold

# Core components
from .scaffold import BasicBlock, ScaffoldNet

# Main functions
from .train import create_model_from_config
from .utils import append_results, format_bytes, load_results, safe_import

__all__ = [
    # Core classes
    "ScaffoldNet",
    "BasicBlock",
    # Configuration
    "ExperimentConfig",
    "BenchmarkConfig",
    "get_dataset_config",
    # Profiling functions
    "get_model_profile",
    "profile_block_in_scaffold",
    # Benchmarking functions
    "benchmark_model",
    "benchmark_block_in_scaffold",
    # Utilities
    "safe_import",
    "append_results",
    "format_bytes",
    "load_results",
    # Training
    "create_model_from_config",
    # Version info
    "__version__",
]


def get_version() -> str:
    """Get the current version of BlockZoo."""
    return __version__


def list_supported_positions() -> list:
    """Get list of supported scaffold positions."""
    return ["early", "mid", "late"]


def list_supported_datasets() -> list:
    """Get list of supported datasets."""
    return ["cifar10", "cifar100", "imagenet"]


# Package-level convenience functions
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
    >>> results = blockzoo.quick_profile('blockzoo.scaffold.BasicBlock', 'mid')
    >>> print(f"Parameters: {results['params_total']}")
    """
    return profile_block_in_scaffold(block_qualified_name=block_class_name, position=position)


def quick_benchmark(block_class_name: str, position: str = "mid") -> dict:
    """
    Quick benchmarking of a block in ScaffoldNet.

    Parameters
    ----------
    block_class_name : str
        Fully qualified name of the block class.
    position : str, optional
        Position to place the block ('early', 'mid', 'late'). Default is 'mid'.

    Returns
    -------
    dict
        Benchmark results.

    Examples
    --------
    >>> results = blockzoo.quick_benchmark('blockzoo.scaffold.BasicBlock', 'mid')
    >>> print(f"Latency: {results['latency_ms']:.2f} ms")
    """
    return benchmark_block_in_scaffold(block_qualified_name=block_class_name, position=position)


# Print package info when imported
def _print_info():
    """Print package information when imported."""
    print(f"[BlockZoo] {__description__}")
    print(f"[BlockZoo] Version: {__version__}")
    print(f"[BlockZoo] Supported positions: {', '.join(list_supported_positions())}")
    print(f"[BlockZoo] Supported datasets: {', '.join(list_supported_datasets())}")


# Only print info in interactive environments, not during tests or scripts
import sys

if hasattr(sys, "ps1") or hasattr(sys, "ps2"):
    _print_info()
