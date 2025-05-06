# Single Image Super-Resolution using GANs

This project implements and evaluates various improvements for Single Image Super-Resolution (SISR) using Generative Adversarial Networks (GANs) on the CelebA dataset.

## Project Structure

```
.
├── configs/                 # Hydra configuration files
├── data/                   # Data loading and preprocessing
├── models/                 # Model architectures
├── utils/                  # Utility functions
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── setup.sh              # Setup script
└── requirements.txt       # Project dependencies
```

## Setup

### Option 1: Automatic Setup (Recommended)

Run the setup script to automatically create and configure the virtual environment:

```bash
chmod +x setup.sh
./setup.sh
```

### Option 2: Manual Setup

1. Create and activate virtual environment:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download dataset:
```bash
gdown 1lzyxD9PCuglTJAuTiDeYbeDO-tEEzRpy
```

## Usage

1. Activate the virtual environment (if not already activated):
```bash
# On Linux/Mac:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

2. Train the model:
```bash
python train.py
```

3. Evaluate the model:
```bash
python evaluate.py
```

4. Deactivate the virtual environment when done:
```bash
deactivate
```

## Features

- Flexible configuration using Hydra
- PyTorch Lightning training pipeline
- GPU acceleration
- Comprehensive metrics (PSNR, SSIM)
- Automatic visualization of results
- Performance tracking and logging

## Virtual Environment Management

- The project uses Python virtual environment to manage dependencies
- All required packages are listed in `requirements.txt`
- The virtual environment is stored in the `venv` directory
- Always activate the virtual environment before running the project
- Use `deactivate` to exit the virtual environment when done 