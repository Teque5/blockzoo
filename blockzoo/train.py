"""Training and evaluation pipeline for BlockZoo framework.

This module provides the main training pipeline that integrates profiling,
benchmarking, and Lightning-based training for convolutional blocks.
"""

import os
import sys
import time
import warnings
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import lightning as L
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from torch import nn
from torch.utils.data import DataLoader

from .benchmark import benchmark_model, print_benchmark_results
from .config import ExperimentConfig, get_dataset_config, parse_train_args, validate_config
from .profile import get_model_profile, print_profile
from .scaffold import ScaffoldNet
from .utils import append_results
from .wrappers import get_block_class

# suppress Lightning warnings for cleaner output
# warnings.filterwarnings("ignore", ".*does not have many workers.*")
# warnings.filterwarnings("ignore", ".*The dataloader.*")


class BlockZooLightningModule(L.LightningModule):
    """
    Lightning module for training blocks in ScaffoldNet.

    Parameters
    ----------
    model : torch.nn.Module
        The ScaffoldNet model to train.
    learning_rate : float
        Learning rate for the optimizer.
    """

    def __init__(self, model: nn.Module, learning_rate: float = 0.001):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)

        # calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).float() / len(y)

        # log metrics
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)

        # calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == y).float() / len(y)

        # log metrics
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}


def create_data_loaders(dataset_name: str, batch_size: int, num_workers: int = 8) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset ('cifar10', 'cifar100', 'imagenet').
    batch_size : int
        Batch size for data loaders.
    num_workers : int, optional
        Number of workers for data loading.

    Returns
    -------
    tuple of DataLoader
        Train and validation data loaders.

    Raises
    ------
    ValueError
        If dataset is not supported.
    """
    dataset_config = get_dataset_config(dataset_name)
    mean = dataset_config["mean"]
    std = dataset_config["std"]

    # define transforms
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4) if dataset_name.startswith("cifar") else transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    # create datasets
    if dataset_name == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root="./data", train=True, download=True, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR10(root="./data", train=False, download=True, transform=val_transform)
    elif dataset_name == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=train_transform)
        val_dataset = torchvision.datasets.CIFAR100(root="./data", train=False, download=True, transform=val_transform)
    elif dataset_name == "imagenet":
        # for ImageNet, we'd need to specify the path - this is a placeholder
        raise NotImplementedError("ImageNet support requires manual dataset setup")
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True if torch.cuda.is_available() else False)

    return train_loader, val_loader


def create_model_from_config(config: ExperimentConfig) -> ScaffoldNet:
    """
    Create ScaffoldNet model from configuration.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.

    Returns
    -------
    ScaffoldNet
        Configured model.
    """
    # get the block class from registry
    block_cls = get_block_class(config.block_class)

    # create scaffolded model
    model = ScaffoldNet(block_cls=block_cls, position=config.position, num_blocks=config.num_blocks, base_channels=config.base_channels, out_dim=config.out_dim)

    return model


def run_training(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run the complete training pipeline.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.

    Returns
    -------
    dict
        Training results including metrics and timing.
    """
    print(f"[BlockZoo] Starting training experiment")
    print(f"  Block: {config.block_class}")
    print(f"  Position: {config.position}")
    print(f"  Dataset: {config.dataset}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Device: {config.device}")

    # create model
    model = create_model_from_config(config)

    # profile model
    print(f"\n[BlockZoo] Profiling model...")
    profile_results = get_model_profile(model, config.input_shape, config.device)
    print_profile(profile_results, f"{config.block_class} (position={config.position})")

    # create data loaders
    print(f"\n[BlockZoo] Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config.dataset, config.batch_size)

    # create Lightning module
    lightning_module = BlockZooLightningModule(model, config.learning_rate)

    # create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        verbose=False,
        dirpath="checkpoints",
        filename=f"{config.block_class}_{config.position}_{config.dataset}_best" + "_{epoch:02d}",
    )

    callbacks = [EarlyStopping(monitor="val_loss", patience=10, verbose=False), checkpoint_callback]

    # configure trainer
    trainer_kwargs = {
        "max_epochs": config.epochs,
        "callbacks": callbacks,
        "enable_progress_bar": True,
        "enable_model_summary": False,
        "logger": False,  # disable logging for simplicity
        "enable_checkpointing": True,  # enable checkpointing to save best model
    }

    # set accelerator
    if config.device == "cuda" and torch.cuda.is_available():
        trainer_kwargs.update({"accelerator": "gpu", "devices": 1})
        # set tensor core precision if available
        torch.set_float32_matmul_precision("high")
    else:
        trainer_kwargs.update({"accelerator": "cpu"})

    trainer = L.Trainer(**trainer_kwargs)

    # train model
    print(f"\n[BlockZoo] Training model...")
    start_time = time.time()

    trainer.fit(lightning_module, train_loader, val_loader)

    training_time = time.time() - start_time

    # load best model checkpoint
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        print(f"[BlockZoo] Loading best model from: {best_model_path}")
        lightning_module = BlockZooLightningModule.load_from_checkpoint(
            best_model_path, model=model, learning_rate=config.learning_rate  # pass the original model architecture
        )
        model = lightning_module.model

    # get final metrics
    train_metrics = trainer.callback_metrics
    val_loss = float(train_metrics.get("val_loss", 0.0))
    val_acc = float(train_metrics.get("val_acc", 0.0))

    print(f"\n[BlockZoo] Training completed in {training_time:.2f} seconds")
    print(f"  Final validation loss: {val_loss:.4f}")
    print(f"  Final validation accuracy: {val_acc:.4f}")

    # prepare results with consistent field ordering
    results = {
        "timestamp": datetime.now().isoformat(),
        "block": config.block_class,
        "dataset": config.dataset,
        "position": config.position,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "lr": config.learning_rate,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "training_time": training_time,
        "params_total": profile_results["params_total"],
        "params_trainable": profile_results["params_trainable"],
        "flops": profile_results["flops"],
        "memory_mb": profile_results["memory_mb"],
        "latency_ms": None,  # will be filled from benchmark if requested
        "latency_std": None,  # will be filled from benchmark if requested
        "throughput": None,  # will be filled from benchmark if requested
        "device": config.device,
        "num_blocks": config.num_blocks,
        "base_channels": config.base_channels,
        "out_dim": config.out_dim,
        "experiment_name": config.experiment_name or "",
        "notes": config.notes or "",
    }

    return results, model


def run_benchmark(model: nn.Module, config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run benchmarking on the model.

    Parameters
    ----------
    model : torch.nn.Module
        Model to benchmark.
    config : ExperimentConfig
        Experiment configuration.

    Returns
    -------
    dict
        Benchmark results.
    """
    print(f"\n[BlockZoo] Running benchmark...")

    benchmark_results = benchmark_model(
        model=model,
        input_shape=config.input_shape,
        device=config.device,
        batch_size=config.batch_size,
        warmup_runs=10,
        benchmark_runs=50,  # fewer runs for training pipeline
    )

    print_benchmark_results(benchmark_results, f"{config.block_class} (position={config.position})")

    return benchmark_results


def main() -> None:
    """Main entry point for blockzoo-train command."""
    try:
        # parse configuration
        config = parse_train_args()
        validate_config(config)

        # full training pipeline
        results, trained_model = run_training(config)

        # run benchmarking (always done)
        benchmark_results = run_benchmark(trained_model, config)
        # add benchmark results to main results
        results.update(
            {"latency_ms": benchmark_results["latency_ms"], "latency_std": benchmark_results["latency_std"], "throughput": benchmark_results["throughput"]}
        )

        # save results to CSV
        append_results(config.output_file, results)
        print(f"\n[BlockZoo] Results saved to {config.output_file}")

        print(f"\n[BlockZoo] Experiment completed successfully!")

    except KeyboardInterrupt:
        print(f"\n[BlockZoo] Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[BlockZoo] Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
