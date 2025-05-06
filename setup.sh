#!/bin/bash

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/celeba
mkdir -p outputs
mkdir -p results

# Download CelebA dataset if not exists
if [ ! -d "data/celeba/img_align_celeba" ]; then
    echo "Downloading CelebA dataset..."
    # Create data directory if it doesn't exist
    mkdir -p data/celeba
    
    # Download dataset
    gdown 1lzyxD9PCuglTJAuTiDeYbeDO-tEEzRpy -O data/celeba/img_align_celeba.zip
    
    # Unzip dataset
    echo "Extracting dataset..."
    unzip data/celeba/img_align_celeba.zip -d data/celeba/
    
    # Clean up
    rm data/celeba/img_align_celeba.zip
fi

echo "Setup completed successfully!"
echo "To activate the virtual environment, run: source venv/bin/activate" 