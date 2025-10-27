"""Tests for blockzoo.train module."""

import unittest
from unittest.mock import MagicMock, patch

from blockzoo.config import ExperimentConfig
from blockzoo.scaffold import BasicBlock, ScaffoldNet
from blockzoo.train import BlockZooLightningModule, create_model_from_config


class TestTrain(unittest.TestCase):
    """Test cases for training module."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ExperimentConfig(block_class="ResNetBasicBlock", position="mid", dataset="cifar10", epochs=1, batch_size=2)

    def test_create_model_from_config(self):
        """Test model creation from configuration."""
        model = create_model_from_config(self.config)

        # check that model is correctly created
        self.assertIsInstance(model, ScaffoldNet)
        self.assertEqual(model.position, "mid")
        self.assertEqual(model.num_blocks, 3)  # default
        self.assertEqual(model.out_dim, 10)  # cifar10

    def test_lightning_module_creation(self):
        """Test Lightning module creation."""
        model = create_model_from_config(self.config)
        lightning_module = BlockZooLightningModule(model, learning_rate=0.001)

        # check that Lightning module is properly set up
        self.assertEqual(lightning_module.learning_rate, 0.001)
        self.assertIsNotNone(lightning_module.model)

    def test_lightning_module_forward(self):
        """Test Lightning module forward pass."""
        import torch

        model = create_model_from_config(self.config)
        lightning_module = BlockZooLightningModule(model, learning_rate=0.001)

        # test forward pass
        x = torch.randn(2, 3, 32, 32)
        output = lightning_module.forward(x)

        self.assertEqual(output.shape, (2, 10))
        self.assertTrue(torch.isfinite(output).all())

    def test_configure_optimizers(self):
        """Test optimizer configuration."""
        model = create_model_from_config(self.config)
        lightning_module = BlockZooLightningModule(model, learning_rate=0.001)

        optimizer_config = lightning_module.configure_optimizers()

        # check that optimizer is configured correctly
        self.assertIn("optimizer", optimizer_config)
        self.assertIn("lr_scheduler", optimizer_config)
        self.assertEqual(optimizer_config["monitor"], "val_loss")


if __name__ == "__main__":
    unittest.main()
