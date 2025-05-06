# Single Image Super-Resolution using GANs

This project implements a Single Image Super-Resolution (SISR) model using Generative Adversarial Networks (GANs) on the CelebA dataset.

## Project Structure

```
.
├── configs/                 # Configuration files
│   ├── config.yaml         # Main configuration
│   ├── model/              # Model configurations
│   │   ├── default.yaml    # Default model config
│   │   └── resnet18.yaml   # ResNet18-based model config
│   ├── training/           # Training configurations
│   │   └── default.yaml    # Default training config
│   ├── data/               # Data configurations
│   │   └── default.yaml    # Default data config
│   ├── hardware/           # Hardware configurations
│   │   └── default.yaml    # Default hardware config
│   └── evaluation/         # Evaluation configurations
│       └── default.yaml    # Default evaluation config
├── data/                   # Data processing modules
│   ├── __init__.py
│   ├── dataset.py         # Dataset implementation
│   └── celeba/            # CelebA dataset directory
├── models/                 # Model architecture
│   ├── __init__.py
│   ├── generator.py       # Generator network
│   └── discriminator.py   # Discriminator network
├── utils/                  # Utility functions
│   ├── __init__.py
│   └── metrics.py         # Evaluation metrics
├── outputs/               # Training outputs and checkpoints
├── results/              # Evaluation results and visualizations
├── train.py              # Training script
├── evaluate.py           # Evaluation script
├── requirements.txt      # Project dependencies
├── setup.sh             # Environment setup script
└── README.md            # Project documentation
```

## Setup

### Automatic Setup (Recommended)

Run the setup script to automatically:
1. Create and configure the virtual environment
2. Install all dependencies
3. Download and prepare the CelebA dataset
4. Create necessary directories

```bash
chmod +x setup.sh
./setup.sh
```

### Manual Setup

If you prefer to set up manually:

1. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download the CelebA dataset:
```bash
# Create data directory
mkdir -p data/celeba

# Download dataset
gdown 1lzyxD9PCuglTJAuTiDeYbeDO-tEEzRpy -O data/celeba/img_align_celeba.zip

# Extract dataset
unzip data/celeba/img_align_celeba.zip -d data/celeba/

# Clean up
rm data/celeba/img_align_celeba.zip
```

## Configuration

This project uses Hydra for configuration management. The configuration is organized into several components:

### Main Configuration
The main configuration file (`configs/config.yaml`) includes:
- Model architecture selection
- Training parameters
- Data settings
- Hardware settings

### Model Configurations
Located in `configs/model/`:
- `default.yaml`: Default model configuration
- `resnet18.yaml`: ResNet18-based model configuration

### Training Configurations
Located in `configs/training/`:
- Learning rate and optimizer settings
- Loss weights
- Scheduler configuration

### Data Configurations
Located in `configs/data/`:
- Dataset parameters
- Data augmentation settings
- DataLoader configuration

### Hardware Configurations
Located in `configs/hardware/`:
- Device selection (CPU/GPU)
- Mixed precision settings
- Distributed training options

### Evaluation Configurations
Located in `configs/evaluation/`:
- Checkpoint selection
- Metrics thresholds
- Visualization settings

### Using Different Configurations

You can override configurations using command-line arguments:

```bash
# Use ResNet18 model configuration
python train.py model=resnet18

# Override specific parameters
python train.py model=resnet18 training.learning_rate=0.0002

# Use different hardware settings
python train.py hardware.device=cpu

# Evaluate with specific checkpoint
python evaluate.py evaluation.checkpoint_path=outputs/checkpoint_epoch_50.pth
```

## Training

To train the model with default configuration:
```bash
python train.py
```

To train with specific configuration:
```bash
python train.py model=resnet18 training.max_epochs=200
```

## Evaluation

To evaluate the model:
```bash
python evaluate.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Features

- Flexible configuration using Hydra
- PyTorch-based training pipeline
- GPU acceleration support
- Comprehensive metrics (PSNR, SSIM)
- Automatic visualization of results
- Performance tracking and logging

## Virtual Environment Management

- The project uses Python virtual environment to manage dependencies
- All required packages are listed in `requirements.txt`
- The virtual environment is stored in the `venv` directory
- Always activate the virtual environment before running the project:
  ```bash
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
- Use `deactivate` to exit the virtual environment when done 