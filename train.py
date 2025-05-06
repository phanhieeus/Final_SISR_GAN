import os
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader

from data.dataset import SISRDataset
from models.generator import Generator
from models.discriminator import Discriminator
from utils.metrics import calculate_metrics
from utils.visualize import visualize_comparison, plot_metrics

class SISRGAN(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(config)
        
        # Initialize models
        self.generator = Generator(
            num_residual_blocks=config.model.generator.num_residual_blocks,
            num_filters=config.model.generator.num_filters,
            use_attention=config.model.generator.use_attention
        )
        self.discriminator = Discriminator(
            num_filters=config.model.discriminator.num_filters,
            num_layers=config.model.discriminator.num_layers,
            use_spectral_norm=config.model.discriminator.use_spectral_norm
        )
        
        # Loss functions
        self.criterion_mse = torch.nn.MSELoss()
        self.criterion_bce = torch.nn.BCEWithLogitsLoss()
        
        # Metrics tracking
        self.train_psnr = []
        self.train_ssim = []
        self.val_psnr = []
        self.val_ssim = []

    def forward(self, x):
        return self.generator(x)

    def training_step(self, batch, batch_idx, optimizer_idx):
        lr_imgs = batch['low_res']
        hr_imgs = batch['high_res']
        
        # Train Generator
        if optimizer_idx == 0:
            # Generate super-resolved images
            sr_imgs = self.generator(lr_imgs)
            
            # Calculate losses
            content_loss = self.criterion_mse(sr_imgs, hr_imgs)
            adversarial_loss = self.criterion_bce(
                self.discriminator(sr_imgs),
                torch.ones_like(self.discriminator(sr_imgs))
            )
            
            # Total generator loss
            g_loss = (
                self.config.training.lambda_content * content_loss +
                self.config.training.lambda_adversarial * adversarial_loss
            )
            
            # Calculate metrics
            psnr, ssim = calculate_metrics(sr_imgs, hr_imgs)
            self.train_psnr.append(psnr)
            self.train_ssim.append(ssim)
            
            self.log('train/g_loss', g_loss)
            self.log('train/psnr', psnr)
            self.log('train/ssim', ssim)
            
            return g_loss
        
        # Train Discriminator
        if optimizer_idx == 1:
            sr_imgs = self.generator(lr_imgs).detach()
            
            # Calculate losses
            real_loss = self.criterion_bce(
                self.discriminator(hr_imgs),
                torch.ones_like(self.discriminator(hr_imgs))
            )
            fake_loss = self.criterion_bce(
                self.discriminator(sr_imgs),
                torch.zeros_like(self.discriminator(sr_imgs))
            )
            
            # Total discriminator loss
            d_loss = (real_loss + fake_loss) / 2
            
            self.log('train/d_loss', d_loss)
            
            return d_loss

    def validation_step(self, batch, batch_idx):
        lr_imgs = batch['low_res']
        hr_imgs = batch['high_res']
        
        # Generate super-resolved images
        sr_imgs = self.generator(lr_imgs)
        
        # Calculate metrics
        psnr, ssim = calculate_metrics(sr_imgs, hr_imgs)
        self.val_psnr.append(psnr)
        self.val_ssim.append(ssim)
        
        self.log('val/psnr', psnr)
        self.log('val/ssim', ssim)
        
        # Visualize results
        if batch_idx == 0:
            save_path = os.path.join(
                self.config.logging.save_dir,
                f'comparison_epoch_{self.current_epoch}.png'
            )
            visualize_comparison(lr_imgs[0], sr_imgs[0], hr_imgs[0], save_path)

    def on_validation_epoch_end(self):
        # Plot metrics
        metrics_path = os.path.join(
            self.config.logging.save_dir,
            f'metrics_epoch_{self.current_epoch}.png'
        )
        plot_metrics(self.val_psnr, self.val_ssim, metrics_path)
        
        # Reset metrics
        self.val_psnr = []
        self.val_ssim = []

    def configure_optimizers(self):
        g_optimizer = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.config.training.learning_rate,
            betas=(self.config.training.beta1, self.config.training.beta2)
        )
        d_optimizer = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.config.training.learning_rate,
            betas=(self.config.training.beta1, self.config.training.beta2)
        )
        return [g_optimizer, d_optimizer]

@hydra.main(config_path="configs", config_name="config")
def main(config: DictConfig):
    # Set random seed
    pl.seed_everything(42)
    
    # Create datasets
    train_dataset = SISRDataset(
        config.data.train_dir,
        high_res_size=config.data.high_res_size,
        low_res_size=config.data.low_res_size
    )
    val_dataset = SISRDataset(
        config.data.val_dir,
        high_res_size=config.data.high_res_size,
        low_res_size=config.data.low_res_size
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers
    )
    
    # Initialize model
    model = SISRGAN(config)
    
    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config.logging.save_dir,
        filename='sisrgan-{epoch:02d}-{val_psnr:.2f}',
        monitor=config.logging.monitor,
        mode=config.logging.mode,
        save_top_k=config.logging.save_top_k
    )
    
    # Create logger
    logger = TensorBoardLogger(
        save_dir=config.logging.save_dir,
        name='sisrgan'
    )
    
    # Create trainer
    trainer = pl.Trainer(
        max_epochs=config.training.max_epochs,
        accelerator=config.hardware.accelerator,
        devices=config.hardware.devices,
        precision=config.hardware.precision,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=config.logging.log_every_n_steps
    )
    
    # Train model
    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main() 