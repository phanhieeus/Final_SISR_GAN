import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np

def calculate_psnr(img1, img2):
    """Calculate PSNR between two images."""
    # Convert to numpy arrays
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    
    # Ensure values are in [0, 1] range
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    
    # Calculate PSNR
    return psnr(img1, img2, data_range=1.0)

def calculate_ssim(img1, img2):
    """Calculate SSIM between two images."""
    # Convert to numpy arrays
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    
    # Ensure values are in [0, 1] range
    img1 = np.clip(img1, 0, 1)
    img2 = np.clip(img2, 0, 1)
    
    # Calculate SSIM
    return ssim(img1, img2, data_range=1.0, channel_axis=0)

def calculate_metrics(sr_img, hr_img):
    """Calculate both PSNR and SSIM metrics."""
    psnr_value = calculate_psnr(sr_img, hr_img)
    ssim_value = calculate_ssim(sr_img, hr_img)
    return psnr_value, ssim_value 