import os
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform_hr=None, transform_lr=None):
        self.root_dir = root_dir
        self.hr_dir = os.path.join(root_dir, 'high_res')
        self.lr_dir = os.path.join(root_dir, 'low_res')
        self.transform_hr = transform_hr
        self.transform_lr = transform_lr
        self.image_files = [f for f in os.listdir(self.hr_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.image_files = sorted(self.image_files)
        

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        hr_img_path = os.path.join(self.hr_dir, self.image_files[idx])
        lr_img_path = os.path.join(self.lr_dir, self.image_files[idx])
        hr_image = Image.open(hr_img_path).convert('RGB')
        lr_image = Image.open(lr_img_path).convert('RGB')
        hr_image = self.transform_hr(hr_image)
        lr_image = self.transform_lr(lr_image)            
        return hr_image, lr_image