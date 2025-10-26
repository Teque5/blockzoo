"""Configuration management for BlockZoo framework.

This module provides configuration classes and utilities for managing
CLI arguments and experiment settings across the BlockZoo framework.
"""

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ExperimentConfig:
    """
    Configuration for BlockZoo experiments.

    Parameters
    ----------
    block_class : str
        Fully qualified name of the block class to test.
    position : str
        Position to place the block ('early', 'mid', 'late').
    dataset : str
        Dataset name ('cifar10', 'cifar100', 'imagenet').
    epochs : int
        Number of training epochs.
    batch_size : int
        Training batch size.
    learning_rate : float
        Learning rate for optimizer.
    input_shape : tuple of int
        Input tensor shape (batch_size, channels, height, width).
    device : str
        Device to use ('cpu' or 'cuda').
    num_blocks : int
        Number of blocks in the scaffold stage.
    base_channels : int
        Base number of channels for the scaffold.
    out_dim : int
        Output dimension (number of classes).
    profile_only : bool
        If True, only profile the model without training.
    benchmark : bool
        If True, run benchmarking after training.
    output_file : str
        Path to CSV file for saving results.
    """

    # Model configuration
    block_class: str
    position: str = "mid"
    num_blocks: int = 3
    base_channels: int = 64
    out_dim: int = 10

    # Training configuration
    dataset: str = "cifar10"
    epochs: int = 10
    batch_size: int = 32
    learning_rate: float = 0.001

    # System configuration
    device: str = "cpu"
    input_shape: Tuple[int, int, int, int] = (1, 3, 32, 32)

    # Experiment configuration
    profile_only: bool = False
    benchmark: bool = False
    output_file: str = "results/results.csv"

    # Optional metadata
    experiment_name: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        # Validate position
        if self.position not in {"early", "mid", "late"}:
            raise ValueError(f"Invalid position: {self.position}")

        # Validate dataset
        if self.dataset not in {"cifar10", "cifar100", "imagenet"}:
            raise ValueError(f"Unsupported dataset: {self.dataset}")

        # Validate device
        if self.device not in {"cpu", "cuda"}:
            raise ValueError(f"Invalid device: {self.device}")

        # Set output dimension based on dataset if not specified
        if self.out_dim == 10 and self.dataset == "cifar100":
            self.out_dim = 100
        elif self.out_dim == 10 and self.dataset == "imagenet":
            self.out_dim = 1000

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "block_class": self.block_class,
            "position": self.position,
            "dataset": self.dataset,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "input_shape": self.input_shape,
            "device": self.device,
            "num_blocks": self.num_blocks,
            "base_channels": self.base_channels,
            "out_dim": self.out_dim,
            "profile_only": self.profile_only,
            "benchmark": self.benchmark,
            "output_file": self.output_file,
            "experiment_name": self.experiment_name,
            "notes": self.notes,
        }


@dataclass
class BenchmarkConfig:
    """
    Configuration for benchmarking experiments.

    Parameters
    ----------
    warmup_runs : int
        Number of warmup runs before benchmarking.
    benchmark_runs : int
        Number of runs for benchmarking.
    batch_sizes : list of int
        List of batch sizes to test.
    """

    warmup_runs: int = 10
    benchmark_runs: int = 100
    batch_sizes: List[int] = field(default_factory=lambda: [1])


def create_train_parser() -> argparse.ArgumentParser:
    """
    Create argument parser for training CLI.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser for training.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate convolutional blocks in BlockZoo framework", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument("block", help="Fully qualified name of the block class (e.g., 'timm.models.resnet.BasicBlock')")

    # Model configuration
    parser.add_argument("--position", choices=["early", "mid", "late"], default="mid", help="Position to place the block in scaffold")
    parser.add_argument("--num-blocks", type=int, default=3, help="Number of blocks in the scaffold stage")
    parser.add_argument("--base-channels", type=int, default=64, help="Base number of channels for the scaffold")

    # Training configuration
    parser.add_argument("--dataset", choices=["cifar10", "cifar100", "imagenet"], default="cifar10", help="Dataset to use for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", "--learning-rate", type=float, default=0.001, dest="learning_rate", help="Learning rate for optimizer")

    # System configuration
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto", help="Device to use for training and inference")
    parser.add_argument(
        "--input-shape",
        type=int,
        nargs=4,
        default=[1, 3, 32, 32],
        metavar=("B", "C", "H", "W"),
        help="Input tensor shape (batch_size, channels, height, width)",
    )

    # Experiment modes
    parser.add_argument("--profile-only", action="store_true", help="Only profile the model without training")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarking after training")

    # Output configuration
    parser.add_argument("--output", default="results/results.csv", help="CSV file to save results")
    parser.add_argument("--experiment-name", help="Name for this experiment (for tracking)")
    parser.add_argument("--notes", help="Additional notes for this experiment")

    # Benchmark configuration (when --benchmark is used)
    parser.add_argument("--warmup-runs", type=int, default=10, help="Number of warmup runs for benchmarking")
    parser.add_argument("--benchmark-runs", type=int, default=100, help="Number of benchmark runs")

    return parser


def parse_train_args(args: Optional[List[str]] = None) -> ExperimentConfig:
    """
    Parse command line arguments for training.

    Parameters
    ----------
    args : list of str, optional
        Arguments to parse. If None, uses sys.argv.

    Returns
    -------
    ExperimentConfig
        Parsed configuration object.
    """
    parser = create_train_parser()
    parsed_args = parser.parse_args(args)

    # Handle auto device selection
    device = parsed_args.device
    if device == "auto":
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Set output dimension based on dataset
    out_dim = 10
    if parsed_args.dataset == "cifar100":
        out_dim = 100
    elif parsed_args.dataset == "imagenet":
        out_dim = 1000

    return ExperimentConfig(
        block_class=parsed_args.block,
        position=parsed_args.position,
        num_blocks=parsed_args.num_blocks,
        base_channels=parsed_args.base_channels,
        dataset=parsed_args.dataset,
        epochs=parsed_args.epochs,
        batch_size=parsed_args.batch_size,
        learning_rate=parsed_args.learning_rate,
        device=device,
        input_shape=tuple(parsed_args.input_shape),
        profile_only=parsed_args.profile_only,
        benchmark=parsed_args.benchmark,
        output_file=parsed_args.output,
        experiment_name=parsed_args.experiment_name,
        notes=parsed_args.notes,
        out_dim=out_dim,
    )


def get_dataset_config(dataset_name: str) -> Dict[str, Any]:
    """
    Get configuration parameters for a dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset ('cifar10', 'cifar100', 'imagenet').

    Returns
    -------
    dict
        Dictionary containing dataset configuration.
    """
    configs = {
        "cifar10": {
            "num_classes": 10,
            "input_size": (3, 32, 32),
            "mean": [0.4914, 0.4822, 0.4465],
            "std": [0.2023, 0.1994, 0.2010],
        },
        "cifar100": {
            "num_classes": 100,
            "input_size": (3, 32, 32),
            "mean": [0.5071, 0.4867, 0.4408],
            "std": [0.2675, 0.2565, 0.2761],
        },
        "imagenet": {
            "num_classes": 1000,
            "input_size": (3, 224, 224),
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
    }

    if dataset_name not in configs:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return configs[dataset_name]


def validate_config(config: ExperimentConfig) -> None:
    """
    Validate experiment configuration.

    Parameters
    ----------
    config : ExperimentConfig
        Configuration to validate.

    Raises
    ------
    ValueError
        If configuration is invalid.
    """
    # Check if output directory exists
    output_path = Path(config.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Validate CUDA availability if requested
    if config.device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise ValueError("CUDA device requested but not available")

    # Validate input shape matches dataset expectations
    dataset_config = get_dataset_config(config.dataset)
    expected_input_size = dataset_config["input_size"]

    # Check channel and spatial dimensions (ignore batch size)
    if config.input_shape[1:] != expected_input_size:
        print(f"[BlockZoo] Warning: Input shape {config.input_shape[1:]} doesn't match " f"expected {expected_input_size} for {config.dataset}")

    # Validate output dimension
    if config.out_dim != dataset_config["num_classes"]:
        print(f"[BlockZoo] Warning: Output dimension {config.out_dim} doesn't match " f"expected {dataset_config['num_classes']} for {config.dataset}")
