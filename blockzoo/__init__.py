"""BlockZoo: Convolutional Block Benchmarking Framework"""

__version__ = "0.2.0"

from .benchmark import benchmark_block_in_scaffold, benchmark_model
from .config import BenchmarkConfig, ExperimentConfig, get_dataset_config
from .profiler import get_model_profile, profile_block_in_scaffold
from .scaffold import ScaffoldNet
from .train import create_model_from_config
from .utils import append_results, format_bytes, load_results

__all__ = [
    # core classes
    "ScaffoldNet",
    # configuration
    "ExperimentConfig",
    "BenchmarkConfig",
    "get_dataset_config",
    # profiling functions
    "get_model_profile",
    "profile_block_in_scaffold",
    # benchmarking functions
    "benchmark_model",
    "benchmark_block_in_scaffold",
    # utilities
    "append_results",
    "format_bytes",
    "load_results",
    # training
    "create_model_from_config",
    # version info
    "__version__",
]


def list_supported_positions() -> list:
    """Get list of supported scaffold positions."""
    return ["early", "mid", "late"]


def list_supported_datasets() -> list:
    """Get list of supported datasets."""
    return ["cifar10", "cifar100", "imagenet"]
