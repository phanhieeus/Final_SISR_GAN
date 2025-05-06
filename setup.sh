#!/bin/bash

# Create virtual environment
echo "Creating Python virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "Installing project dependencies..."
pip install -r requirements.txt

# Download dataset
echo "Downloading dataset..."
gdown 1lzyxD9PCuglTJAuTiDeYbeDO-tEEzRpy

# Create necessary directories
echo "Creating project directories..."
mkdir -p logs
mkdir -p super_resolution_dataset/{train,val,test}/{high_res,low_res}

echo "Setup completed! To activate the environment, run: source venv/bin/activate" 