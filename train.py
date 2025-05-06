from tqdm import tqdm
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import os
from dataset import ImageDataset
from generator import Generator
from discriminator import Discriminator
from loss import GeneratorLoss, DiscriminatorLoss
from loss import VGGExtractor


def train(generator, discriminator, epochs=20, batch_size=16, lr=1e-5, save_freq=5, checkpoint_dir='checkpoints', resume_from=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on {device}")
    
    # Tạo thư mục checkpoints nếu chưa tồn tại
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Khởi tạo VGG extractor cho perceptual loss
    vgg_extractor = VGGExtractor(6).to(device)
    vgg_extractor.eval()  # Đặt VGG ở chế độ đánh giá
    
    # Định nghĩa các phép biến đổi cho ảnh HR và LR
    transform_hr = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    transform_lr = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Khởi tạo dataset và dataloader
    dataset = ImageDataset(root_dir='./super_resolution_dataset/train',
                           transform_hr=transform_hr, transform_lr=transform_lr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                           num_workers=4, pin_memory=True)  # Tối ưu dataloader

    # Đưa các mô hình lên thiết bị
    generator = generator.to(device)
    discriminator = discriminator.to(device)
    
    # Khởi tạo các hàm loss
    generator_loss = GeneratorLoss(vgg_extractor).to(device)
    discriminator_loss = DiscriminatorLoss().to(device)

    # Khởi tạo optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.9, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.9, 0.999))
    
    # Tạo scheduler để giảm learning rate theo thời gian
    scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(optimizer_g, mode='min', factor=0.5, patience=5)
    scheduler_d = optim.lr_scheduler.ReduceLROnPlateau(optimizer_d, mode='min', factor=0.5, patience=5)
    
    # Khôi phục từ checkpoint nếu có
    start_epoch = 0
    if resume_from:
        if os.path.exists(resume_from):
            checkpoint = torch.load(resume_from)
            generator.load_state_dict(checkpoint['generator'])
            discriminator.load_state_dict(checkpoint['discriminator'])
            optimizer_g.load_state_dict(checkpoint['optimizer_g'])
            optimizer_d.load_state_dict(checkpoint['optimizer_d'])
            start_epoch = checkpoint['epoch']
            print(f"Resumed training from epoch {start_epoch}")
        else:
            print(f"Checkpoint {resume_from} not found. Starting from scratch.")

    # Vòng lặp huấn luyện
    for ep in range(start_epoch, start_epoch + epochs):
        # Đặt mô hình ở chế độ huấn luyện
        generator.train()
        discriminator.train()
        
        loop = tqdm(dataloader, leave=True, desc=f"Epoch [{ep+1}/{start_epoch + epochs}]")

        total_d_loss = 0.0
        total_g_loss = 0.0
        num_batches = 0

        for hr, lr in loop:
            batch_size = hr.size(0)  # Kích thước batch thực tế
            hr, lr = hr.to(device), lr.to(device)

            # ----------------------
            # Huấn luyện Discriminator
            # ----------------------
            optimizer_d.zero_grad()
            
            # Tạo ảnh super-resolution
            with torch.no_grad():  # Không tính gradient cho generator khi huấn luyện discriminator
                sr_imgs = generator(lr)
            
            # Dự đoán của discriminator
            real_pred = discriminator(hr)
            fake_pred = discriminator(sr_imgs.detach())  # Tách sr_imgs khỏi đồ thị tính toán
            
            # Tính loss cho discriminator
            loss_d = discriminator_loss(fake_pred, real_pred)
            
            # Backpropagation
            loss_d.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)  # Ngăn gradient exploding
            optimizer_d.step()

            # ----------------------
            # Huấn luyện Generator
            # ----------------------
            optimizer_g.zero_grad()
            
            # Tạo lại ảnh super-resolution (cho phép tính gradient)
            sr_imgs = generator(lr)
            
            # Dự đoán mới từ discriminator
            fake_pred = discriminator(sr_imgs)
            
            # Tính loss cho generator
            loss_g = generator_loss(sr_imgs, hr, fake_pred, real_pred)
            
            # Backpropagation
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)  # Ngăn gradient exploding
            optimizer_g.step()

            # Cập nhật thống kê
            total_d_loss += loss_d.item()
            total_g_loss += loss_g.item()
            num_batches += 1

            # Cập nhật thanh tiến trình
            loop.set_postfix(D_loss=f"{loss_d.item():.4f}", G_loss=f"{loss_g.item():.4f}")

        # Tính loss trung bình cho epoch
        avg_d_loss = total_d_loss / num_batches
        avg_g_loss = total_g_loss / num_batches
        
        # Cập nhật learning rate dựa trên loss
        scheduler_g.step(avg_g_loss)
        scheduler_d.step(avg_d_loss)
        
        # In thông tin về learning rate hiện tại
        current_lr_g = optimizer_g.param_groups[0]['lr']
        current_lr_d = optimizer_d.param_groups[0]['lr']
        print(f"Epoch {ep+1} finished. Avg D_loss: {avg_d_loss:.4f}, Avg G_loss: {avg_g_loss:.4f}")
        print(f"Current learning rates - Generator: {current_lr_g:.1e}, Discriminator: {current_lr_d:.1e}")

        # Lưu checkpoint theo định kỳ
        if (ep + 1) % save_freq == 0:
            checkpoint = {
                'epoch': ep + 1,
                'generator': generator.state_dict(),
                'discriminator': discriminator.state_dict(),
                'optimizer_g': optimizer_g.state_dict(),
                'optimizer_d': optimizer_d.state_dict(),
                'g_loss': avg_g_loss,
                'd_loss': avg_d_loss
            }
            torch.save(checkpoint, f'{checkpoint_dir}/checkpoint_epoch_{ep+1}.pth')
            print(f"Checkpoint saved at epoch {ep+1}")

    print("Training completed.")



def main():
    # Khởi tạo generator và discriminator
    generator = Generator()
    discriminator = Discriminator()
    
    # Bắt đầu huấn luyện
    train(generator, discriminator, epochs=20, batch_size=16, lr=1e-4)


if __name__ == "__main__":
    main()