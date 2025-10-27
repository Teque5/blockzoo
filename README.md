# BlockZoo: Convolutional Block Benchmarking Framework

Given the demise of benchmarks from paperswithcode and the ongoing need to evaluate convolutional building blocks, BlockZoo provides a standardized framework to benchmark and profile these blocks in isolation. By embedding blocks into a fixed scaffold architecture at different positions (early, mid, late), we can measure block specialization and performance.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.0+-792ee5.svg)](https://lightning.ai/)

## Features

- **Scaffold Architecture**: Fixed stem → StageA → StageB → StageC → head for consistent evaluation
- **Positional Analysis**: Test blocks in early/mid/late positions to measure specialization
- **Comprehensive Profiling**: FLOPs, parameters, memory usage, and runtime benchmarking
- **Lightning Integration**: Robust training pipeline with PyTorch Lightning
- **CSV Export**: Structured results for analysis and visualization
- **Dynamic Loading**: Import any block class by qualified name

## Quick Start

### Installation

```bash
git clone git@github.com:Teque5/blockzoo.git
pip install --editable .
```

### Basic Usage

#### Profile a Block (Quick)

```bash
# Profile a ResNet BasicBlock in mid position
python -m blockzoo.profiler ResNetBasicBlock --position mid

# Output:
# [BlockZoo] Profile for ResNetBasicBlock (position=mid):
#   Parameters (total):     74,058
#   Parameters (trainable): 74,058
#   FLOPs:                  29,490,176
#   Memory estimate:        1.2 MB
```

#### Benchmark Runtime Performance

```bash
python -m blockzoo.benchmark ResNetBasicBlock --position mid --device cuda

# Output:
# [BlockZoo] Benchmark results for ResNetBasicBlock (position=mid):
#   Forward pass: 1.234 ms (avg over 1000 runs)
#   Backward pass: 2.345 ms (avg over 1000 runs)
#   GPU memory: 45.6 MB
```

#### Full Training & Evaluation

```bash
# profile, benchmark, train, and evaluate
```bash
python -m blockzoo.train ResNetBasicBlock --position mid --epochs 5 --benchmark
# see results/results.csv for logged metrics
```

#### Running Comparisons

Compare block performance across positions:

```bash
python -m blockzoo.train ResNetBasicBlock --position early
python -m blockzoo.train ResNetBasicBlock --position mid
python -m blockzoo.train ResNetBasicBlock --position late
```

#### Run all benchmarks

```bash
python3 scripts/bench_all.py
```

## Positional Specialization Protocol

The core innovation of BlockZoo is measuring how blocks perform across different network positions. This protocol helps identify whether blocks are specialized for local features (early layers) or global features (late layers).

### Protocol Overview

1. **Fixed Scaffold**: Use ScaffoldNet with identical stem/head across all experiments
2. **Three Canonical Positions**:
   - **Early** (Stage A): High resolution, small receptive field - focuses on local context (edges, textures)
   - **Mid** (Stage B): Medium resolution with 2x downsampling - balances local and global context
   - **Late** (Stage C): Low resolution with 4x downsampling - emphasizes global context (objects, scenes)
3. **Isolation Testing**: Only one stage active per experiment
4. **Consistent Training**: Same optimizer, schedule, and data across positions

### Position Comparison Example

```bash
# Test ResNet BasicBlock across all positions
python -m blockzoo.train ResNetBasicBlock --position early --epochs 10
python -m blockzoo.train ResNetBasicBlock --position mid --epochs 10
python -m blockzoo.train ResNetBasicBlock --position late --epochs 10

# Analyze results
python -c "
import pandas as pd
df = pd.read_csv('results/results.csv')
print(df.groupby('position')[['val_acc', 'params_total', 'latency_ms']].mean())
"
```

## Detailed Usage

### Command Line Interface

#### Training Command (`blockzoo-train`)

```bash
python -m blockzoo.train <block_class> [options]

# Required:
#   block_class              Fully qualified block class name

# Model Configuration:
#   --position {early,mid,late}    Position in scaffold (default: mid)
#   --num-blocks INT               Number of blocks in stage (default: 3)
#   --base-channels INT            Base channels for scaffold (default: 64)

# Training Configuration:
#   --dataset {cifar10,cifar100}   Dataset to use (default: cifar10)
#   --epochs INT                   Training epochs (default: 10)
#   --batch-size INT               Batch size (default: 32)
#   --lr FLOAT                     Learning rate (default: 0.001)

# System Configuration:
#   --device {cpu,cuda,auto}       Device to use (default: auto)
#   --input-shape B C H W          Input tensor shape (default: 1 3 32 32)

# Modes:
#   --profile-only                 Only profile, skip training
#   --benchmark                    Run benchmarking after training

# Output:
#   --output PATH                  CSV file for results (default: results/results.csv)
#   --experiment-name STR          Name for experiment tracking
#   --notes STR                    Additional notes
```

#### Profiling Command (`blockzoo-profile`)

```bash
python -m blockzoo.profiler <block_class> [options]

# Options:
#   --position {early,mid,late}    Scaffold position (default: mid)
#   --device {cpu,cuda}            Device for profiling (default: cpu)
#   --input-shape B C H W          Input shape (default: 1 3 32 32)
#   --num-blocks INT               Number of blocks (default: 3)
#   --output PATH                  Save results to CSV
```

#### Benchmarking Command (`blockzoo-benchmark`)

```bash
python -m blockzoo.benchmark <block_class> [options]

# Options:
#   --position {early,mid,late}    Scaffold position (default: mid)
#   --device {cpu,cuda}            Device for benchmarking (default: cpu)
#   --batch-size INT               Batch size (default: 1)
#   --warmup-runs INT              Warmup iterations (default: 10)
#   --benchmark-runs INT           Benchmark iterations (default: 100)
#   --multi-batch INT [INT ...]    Test multiple batch sizes
#   --output PATH                  Save results to CSV
```

### Python API

#### Quick Profiling and Benchmarking

```python
import blockzoo

# Quick profiling
profile = blockzoo.quick_profile('ResNetBasicBlock', position='mid')
print(f"Parameters: {profile['params_total']:,}")
print(f"FLOPs: {profile['flops']:,}")

# Quick benchmarking
benchmark = blockzoo.quick_benchmark('ResNetBasicBlock', position='mid')
print(f"Latency: {benchmark['latency_ms']:.2f} ms")
print(f"Throughput: {benchmark['throughput']:.1f} img/s")
```

#### Advanced Usage

```python
import torch
from blockzoo import ScaffoldNet, ExperimentConfig, get_model_profile
from blockzoo.wrappers import get_block_class

# Import and create a custom block
BlockClass = get_block_class('ResNetBasicBlock')
model = ScaffoldNet(BlockClass, position='early', num_blocks=4)

# Profile the model
profile = get_model_profile(model, input_shape=(1, 3, 224, 224))
print(f"Model has {profile['params_total']:,} parameters")

# Create experiment configuration
config = ExperimentConfig(
    block_class='ResNetBasicBlock',
    position='late',
    dataset='cifar100',
    epochs=20,
    batch_size=64
)

# Use the configuration (would typically be done in train.py)
print(f"Config: {config.to_dict()}")
```

#### Custom Block Integration

```python
from torch import nn
from blockzoo import ScaffoldNet

class MyCustomBlock(nn.Module):
    """Custom convolutional block."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return self.relu(out)

# Test your custom block
model = ScaffoldNet(MyCustomBlock, position='mid')
profile = blockzoo.get_model_profile(model)
print(f"Custom block profile: {profile}")
```

## Results Analysis

### CSV Schema

The framework saves results to CSV with the following columns:

```csv
timestamp,block,dataset,position,epochs,batch_size,lr,
val_acc,val_loss,training_time,
params_total,params_trainable,flops,memory_mb,
latency_ms,latency_std,throughput,
device,num_blocks,base_channels,out_dim,
experiment_name,notes
```

### Analysis Examples

#### Load and Analyze Results

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load results
df = pd.read_csv('results/results.csv')

# Position-wise analysis
position_analysis = df.groupby(['block', 'position']).agg({
    'val_acc': 'mean',
    'params_total': 'mean',
    'latency_ms': 'mean',
    'flops': 'mean'
}).round(4)

print(position_analysis)
```

#### Calculate Positional Metrics

```python
# Calculate generality scores for each block
def calculate_generality(group):
    """Calculate generality as MeanPos / (1 + VarPos)."""
    mean_acc = group['val_acc'].mean()
    var_acc = group['val_acc'].var()
    return mean_acc / (1 + var_acc)

generality_scores = df.groupby('block').apply(calculate_generality)
print("Block generality scores:")
print(generality_scores.sort_values(ascending=False))
```

#### Visualization

```python
import seaborn as sns

# Position vs Accuracy plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='position', y='val_acc')
plt.title('Validation Accuracy by Position')
plt.ylabel('Validation Accuracy')
plt.show()

# Performance vs Efficiency scatter
plt.figure(figsize=(10, 6))
plt.scatter(df['params_total'], df['val_acc'],
           c=df['position'].astype('category').cat.codes,
           alpha=0.7)
plt.xlabel('Parameters')
plt.ylabel('Validation Accuracy')
plt.colorbar(label='Position')
plt.show()
```

## Architecture Details

### ScaffoldNet Structure

```
Input (3×32×32)
    ↓
Stem: Conv3x3(3→64) + BN + ReLU
    ↓
Stage A (Early): 64→64, stride=1, high-res
    ↓
Stage B (Mid): 64→128, stride=2, medium-res
    ↓
Stage C (Late): 128→256, stride=2, low-res
    ↓
Head: AdaptiveAvgPool + Linear(256→classes)
    ↓
Output (classes,)
```

**Key Design Choices:**
- **Fixed Stem/Head**: Ensures consistent feature extraction/classification
- **Single Active Stage**: Isolates block performance
- **Progressive Channels**: 64 → 128 → 256 following common practices
- **Controlled Downsampling**: 2× at each transition

### Supported Block Interfaces

Blocks must implement the standard PyTorch signature:

```python
def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
    """
    Parameters:
    - in_channels: Number of input channels
    - out_channels: Number of output channels
    - stride: Convolution stride (1 or 2)
    """
```

**Available Block Types:**

All blocks are defined in the BlockZoo registry and follow the unified (in_channels, out_channels, stride) interface:

- **ResNet blocks**: `ResNetBasicBlock`, `ResNetBottleneck`
- **EfficientNet blocks**: `InvertedResidual`, `UniversalInvertedResidual`, `EdgeResidual`
- **Simple fallback blocks**: `SimpleResidualBlock`
- **Basic BlockZoo blocks**: `BasicBlock`

To see all available blocks:
```python
from blockzoo.wrappers import list_available_blocks
print("Available blocks:", list_available_blocks())
```

## Testing

Run the test suite to verify installation:

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_blockzoo.py::TestScaffoldNet -v
python -m pytest tests/test_blockzoo.py::TestProfiler -v
python -m pytest tests/test_blockzoo.py::TestBenchmark -v

# Run with coverage
python -m pytest tests/ --cov=blockzoo --cov-report=html
```

## Examples

### Example 1: Compare ResNet vs DenseNet Blocks

```bash
# ResNet BasicBlock
python -m blockzoo.train ResNetBasicBlock --position mid --epochs 10 --experiment-name "resnet-basic"

# Simple Residual Block
python -m blockzoo.train SimpleResidualBlock --position mid --epochs 10 --experiment-name "simple-residual"

# Compare results
python -c "
import pandas as pd
df = pd.read_csv('results/results.csv')
comparison = df.groupby('experiment_name')[['val_acc', 'params_total', 'latency_ms']].mean()
print(comparison)
"
```

### Example 2: Position Sensitivity Analysis

```bash
#!/bin/bash
# Test EfficientNet block across positions

BLOCK="InvertedResidual"

for pos in early mid late; do
    echo "Testing position: $pos"
    python -m blockzoo.train $BLOCK \
        --position $pos \
        --epochs 15 \
        --benchmark \
        --experiment-name "efficientnet-${pos}" \
        --notes "Position sensitivity test"
done

echo "Analysis complete. Check results/results.csv"
```

### Example 3: Batch Size Scaling

```bash
# Test how performance scales with batch size
python -m blockzoo.benchmark ResNetBasicBlock \
    --position mid \
    --multi-batch 1 2 4 8 16 32 \
    --device cuda \
    --output results/batch_scaling.csv
```

## Advanced Configuration

### Environment Variables

```bash
# Set default device
export BLOCKZOO_DEVICE=cuda

# Set default results directory
export BLOCKZOO_RESULTS_DIR=./experiments/results

# Disable progress bars (for automated runs)
export BLOCKZOO_QUIET=1
```

### Custom Dataset Integration

```python
# Extend config.py to add custom datasets
def get_custom_dataset_config(dataset_name: str):
    if dataset_name == "my_dataset":
        return {
            "num_classes": 50,
            "input_size": (3, 64, 64),
            "mean": [0.5, 0.5, 0.5],
            "std": [0.25, 0.25, 0.25]
        }
        # ... add more custom datasets
```

# Run tests
python -m pytest tests/ -v --cov=blockzoo
```
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python -m blockzoo.train <block> --batch-size 16

# Use CPU fallback
python -m blockzoo.train <block> --device cpu
```

**2. Import Errors**
```bash
# Install missing dependencies
pip install timm torchinfo fvcore

# Check available block names
python -c "from blockzoo.wrappers import list_available_blocks; print('Available blocks:', list_available_blocks())"
```

**3. Lightning Warnings**
```python
# Suppress in your script
import warnings
warnings.filterwarnings("ignore", ".*dataloader.*")
```

## Contributing

We welcome contributions! Please:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Add tests** for new functionality
4. **Run tests**: `python -m pytest tests/`
5. **Commit** changes (`git commit -m 'Add amazing feature'`)
6. **Push** to branch (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request

### Development Setup

```bash
# Clone and install in development mode
git clone https://github.com/yourusername/blockzoo.git
cd blockzoo
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
python -m pytest tests/ -v --cov=blockzoo
```

---

*For questions, issues, or feature requests, please open an issue on GitHub.*

---

**Happy benchmarking!**
