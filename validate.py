import torch
from tqdm import tqdm


def validate(generator, dataloader, device):
    """Hàm đánh giá mô hình trên tập validation"""
    generator.eval()
    total_psnr = 0
    total_samples = 0
    
    with torch.no_grad():
        for hr, lr in tqdm(dataloader, desc="Validating"):
            hr, lr = hr.to(device), lr.to(device)
            sr = generator(lr)
            
            # Chuyển từ [-1, 1] về [0, 1] để tính PSNR
            hr = (hr + 1) / 2
            sr = (sr + 1) / 2
            
            # Tính PSNR
            mse = torch.mean((hr - sr) ** 2)
            psnr = 10 * torch.log10(1 / mse)
            
            total_psnr += psnr.item()
            total_samples += 1
    
    return total_psnr / total_samples