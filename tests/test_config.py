"""Tests for blockzoo.config module."""

import unittest

from blockzoo.config import ExperimentConfig, get_dataset_config, parse_train_args


class TestConfig(unittest.TestCase):
    """Test cases for configuration module."""

    def test_experiment_config_creation(self):
        """Test ExperimentConfig creation using parse_train_args."""
        config = parse_train_args(["ResNetBasicBlock", "--position", "mid", "--dataset", "cifar10"])

        self.assertEqual(config.block_class, "ResNetBasicBlock")
        self.assertEqual(config.position, "mid")
        self.assertEqual(config.dataset, "cifar10")
        self.assertEqual(config.out_dim, 10)  # should be set based on dataset

    def test_experiment_config_validation(self):
        """Test ExperimentConfig validation."""
        # Test invalid position - should raise error during creation
        with self.assertRaises(ValueError):
            ExperimentConfig(
                block_class="test",
                position="invalid",
                num_blocks=3,
                base_channels=64,
                out_dim=10,
                dataset="cifar10",
                epochs=25,
                batch_size=256,
                learning_rate=0.001,
                device="cpu",
                input_shape=(1, 3, 32, 32),
                output_file="results.csv",
                experiment_name=None,
                notes=None,
            )

        # Test invalid dataset - should raise error during creation
        with self.assertRaises(ValueError):
            ExperimentConfig(
                block_class="test",
                position="mid",
                num_blocks=3,
                base_channels=64,
                out_dim=10,
                dataset="invalid",
                epochs=25,
                batch_size=256,
                learning_rate=0.001,
                device="cpu",
                input_shape=(1, 3, 32, 32),
                output_file="results.csv",
                experiment_name=None,
                notes=None,
            )

    def test_get_dataset_config(self):
        """Test get_dataset_config function."""
        cifar10_config = get_dataset_config("cifar10")

        expected_keys = {"num_classes", "input_size", "mean", "std"}
        self.assertEqual(set(cifar10_config.keys()), expected_keys)
        self.assertEqual(cifar10_config["num_classes"], 10)
        self.assertEqual(cifar10_config["input_size"], (3, 32, 32))

    def test_get_dataset_config_invalid(self):
        """Test get_dataset_config with invalid dataset."""
        with self.assertRaises(ValueError):
            get_dataset_config("invalid_dataset")


if __name__ == "__main__":
    unittest.main()
