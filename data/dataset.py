import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class SISRDataset(Dataset):
    def __init__(self, root_dir, high_res_size=256, low_res_size=64):
        self.root_dir = Path(root_dir)
        self.high_res_dir = self.root_dir / "high_res"
        self.low_res_dir = self.root_dir / "low_res"
        self.high_res_size = high_res_size
        self.low_res_size = low_res_size
        
        self.image_files = sorted([f for f in os.listdir(self.high_res_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        self.high_res_transform = transforms.Compose([
            transforms.Resize((high_res_size, high_res_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.low_res_transform = transforms.Compose([
            transforms.Resize((low_res_size, low_res_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        
        # Load high-res image
        high_res_path = self.high_res_dir / img_name
        high_res_img = Image.open(high_res_path).convert('RGB')
        high_res_tensor = self.high_res_transform(high_res_img)
        
        # Load low-res image
        low_res_path = self.low_res_dir / img_name
        low_res_img = Image.open(low_res_path).convert('RGB')
        low_res_tensor = self.low_res_transform(low_res_img)
        
        return {
            'high_res': high_res_tensor,
            'low_res': low_res_tensor,
            'filename': img_name
        } 