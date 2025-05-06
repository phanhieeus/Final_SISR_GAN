import os
import hydra
from omegaconf import DictConfig
import torch
from torch.utils.data import DataLoader
import pandas as pd
from pathlib import Path

from data.dataset import SISRDataset
from models.generator import Generator
from utils.metrics import calculate_metrics
from utils.visualize import visualize_comparison

@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    # Create test dataset and dataloader
    test_dataset = SISRDataset(
        config.data.test_dir,
        high_res_size=config.data.high_res_size,
        low_res_size=config.data.low_res_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.data.num_workers
    )
    
    # Initialize generator
    generator = Generator(
        num_residual_blocks=config.model.generator.num_residual_blocks,
        num_filters=config.model.generator.num_filters,
        use_attention=config.model.generator.use_attention
    )
    
    # Load best model
    checkpoint_path = os.path.join(config.logging.save_dir, 'best_model.ckpt')
    if os.path.exists(checkpoint_path):
        generator.load_state_dict(torch.load(checkpoint_path)['generator_state_dict'])
    else:
        print("No checkpoint found. Please train the model first.")
        return
    
    # Move model to device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = generator.to(device)
    generator.eval()
    
    # Create results directory
    results_dir = Path(config.logging.save_dir) / 'test_results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize metrics storage
    metrics = {
        'filename': [],
        'psnr': [],
        'ssim': []
    }
    
    # Evaluate on test set
    with torch.no_grad():
        for batch in test_loader:
            lr_imgs = batch['low_res'].to(device)
            hr_imgs = batch['high_res'].to(device)
            filenames = batch['filename']
            
            # Generate super-resolved images
            sr_imgs = generator(lr_imgs)
            
            # Calculate metrics
            psnr, ssim = calculate_metrics(sr_imgs, hr_imgs)
            
            # Store metrics
            metrics['filename'].extend(filenames)
            metrics['psnr'].append(psnr)
            metrics['ssim'].append(ssim)
            
            # Save visualization
            for i in range(len(filenames)):
                save_path = results_dir / f"{filenames[i]}_comparison.png"
                visualize_comparison(
                    lr_imgs[i].cpu(),
                    sr_imgs[i].cpu(),
                    hr_imgs[i].cpu(),
                    save_path
                )
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(results_dir / 'test_metrics.csv', index=False)
    
    # Print summary
    print("\nTest Results Summary:")
    print(f"Average PSNR: {metrics_df['psnr'].mean():.2f} dB")
    print(f"Average SSIM: {metrics_df['ssim'].mean():.4f}")
    print(f"\nResults saved to: {results_dir}")

if __name__ == "__main__":
    main() 