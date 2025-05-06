import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def denormalize(tensor):
    """Convert normalized tensor to image."""
    return (tensor * 0.5 + 0.5).clamp(0, 1)

def visualize_comparison(lr_img, sr_img, hr_img, save_path=None):
    """Visualize comparison between low-res, super-resolved, and high-res images."""
    # Convert tensors to numpy arrays
    lr_img = denormalize(lr_img).cpu().numpy()
    sr_img = denormalize(sr_img).cpu().numpy()
    hr_img = denormalize(hr_img).cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot images
    axes[0].imshow(np.transpose(lr_img, (1, 2, 0)))
    axes[0].set_title('Low Resolution')
    axes[0].axis('off')
    
    axes[1].imshow(np.transpose(sr_img, (1, 2, 0)))
    axes[1].set_title('Super Resolved')
    axes[1].axis('off')
    
    axes[2].imshow(np.transpose(hr_img, (1, 2, 0)))
    axes[2].set_title('High Resolution')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_metrics(psnr_values, ssim_values, save_path=None):
    """Plot PSNR and SSIM values over time."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot PSNR
    ax1.plot(psnr_values)
    ax1.set_title('PSNR over time')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('PSNR (dB)')
    ax1.grid(True)
    
    # Plot SSIM
    ax2.plot(ssim_values)
    ax2.set_title('SSIM over time')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('SSIM')
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 