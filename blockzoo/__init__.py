"""BlockZoo: Convolutional Block Benchmarking Framework"""

__version__ = "0.3.0"

from .config import BenchmarkConfig, ExperimentConfig, get_dataset_config
from .scaffold import ScaffoldNet
from .utils import append_results, format_bytes, load_results

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


def list_supported_positions() -> list:
    """Get list of supported scaffold positions."""
    return ["early", "mid", "late"]


def list_supported_datasets() -> list:
    """Get list of supported datasets."""
    return ["cifar10", "cifar100", "imagenet"]
