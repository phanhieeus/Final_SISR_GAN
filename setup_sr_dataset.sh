#!/bin/bash

# Cập nhật hệ thống
sudo apt update && sudo apt install -y python3-pip python3-venv unzip git wget

# Nâng cấp pip và cài thư viện
pip install --upgrade pip
pip install gdown

# Tải dataset từ Google Drive bằng gdown
gdown 1lzyxD9PCuglTJAuTiDeYbeDO-tEEzRpy

# Giải nén dataset
unzip super_resolution_dataset.zip -d super_resolution_dataset

echo "✅ DONE! Dataset nằm trong thư mục: super_resolution_dataset/"
