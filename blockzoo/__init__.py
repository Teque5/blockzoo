"""BlockZoo: Convolutional Block Benchmarking Framework"""

__version__ = "0.3.0"

from .config import BenchmarkConfig, ExperimentConfig, get_dataset_config
from .scaffold import ScaffoldNet
from .utils import append_results, format_bytes, load_results
from .wrappers import BLOCK_REGISTRY

__all__ = [
    # core classes
    "ScaffoldNet",
    # configuration
    "ExperimentConfig",
    "BenchmarkConfig",
    "get_dataset_config",
    # utilities
    "append_results",
    "format_bytes",
    "load_results",
    # version info
    "__version__",
]


def list_supported_positions() -> list[str]:
    """Get list of supported scaffold positions."""
    return ["early", "mid", "late"]


def list_supported_datasets() -> list[str]:
    """Get list of supported datasets."""
    return ["cifar10", "cifar100", "imagenet"]


def list_available_blocks() -> list[str]:
    """List all available block names."""
    return list(BLOCK_REGISTRY.keys())
