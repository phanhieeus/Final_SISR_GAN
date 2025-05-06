import os
import argparse
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from generator import Generator
from tqdm import tqdm


def preprocess_image(img_path, scale_factor=4):
    """Đọc và tiền xử lý ảnh đầu vào"""
    img = Image.open(img_path).convert('RGB')
    
    # Lưu kích thước gốc để sử dụng sau này
    original_width, original_height = img.size
    
    # Chuyển đổi ảnh đầu vào
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    return transform(img), original_width, original_height


def denormalize(tensor):
    """Chuyển từ [-1, 1] về [0, 1]"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)


def inference_single_image(generator, img_path, output_path, device):
    """Thực hiện super-resolution cho một ảnh đơn lẻ"""
    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Đọc và tiền xử lý ảnh
    img_tensor, original_width, original_height = preprocess_image(img_path)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # Thêm batch dimension
    
    # Dự đoán với mô hình
    with torch.no_grad():
        sr_img = generator(img_tensor)
    
    # Xử lý sau dự đoán
    sr_img = denormalize(sr_img)
    
    # Lưu ảnh kết quả
    save_image(sr_img, output_path)
    print(f"Saved super-resolution image to {output_path}")


def inference_batch(generator, input_dir, output_dir, device):
    """Thực hiện super-resolution cho toàn bộ ảnh trong thư mục"""
    # Tạo thư mục đầu ra nếu chưa có
    os.makedirs(output_dir, exist_ok=True)
    
    # Lấy danh sách tất cả các ảnh trong thư mục đầu vào
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    img_files = [f for f in os.listdir(input_dir) 
                if os.path.splitext(f.lower())[1] in img_extensions]
    
    if not img_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(img_files)} images. Processing...")
    
    # Xử lý từng ảnh
    for img_file in tqdm(img_files):
        input_path = os.path.join(input_dir, img_file)
        output_path = os.path.join(output_dir, f"sr_{img_file}")
        inference_single_image(generator, input_path, output_path, device)


def calculate_psnr(img1, img2):
    """Tính PSNR giữa hai ảnh"""
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def evaluate(generator, test_dir, hr_dir, device):
    """Đánh giá mô hình trên tập test có ground truth"""
    # Lấy danh sách tất cả các ảnh trong thư mục đầu vào
    img_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    lr_files = [f for f in os.listdir(test_dir) 
               if os.path.splitext(f.lower())[1] in img_extensions]
    
    if not lr_files:
        print(f"No image files found in {test_dir}")
        return
    
    total_psnr = 0
    processed_images = 0
    
    print(f"Evaluating {len(lr_files)} test images...")
    
    # Định nghĩa transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    for lr_file in tqdm(lr_files):
        lr_path = os.path.join(test_dir, lr_file)
        
        # Tìm ảnh HR tương ứng
        hr_file = lr_file.replace('LR', 'HR') if 'LR' in lr_file else f"HR_{lr_file}"
        hr_path = os.path.join(hr_dir, hr_file)
        
        if not os.path.exists(hr_path):
            print(f"Warning: HR image {hr_path} not found. Skipping...")
            continue
        
        # Đọc ảnh LR và HR
        lr_img = Image.open(lr_path).convert('RGB')
        hr_img = Image.open(hr_path).convert('RGB')
        
        # Chuyển đổi sang tensor
        lr_tensor = transform(lr_img).unsqueeze(0).to(device)
        hr_tensor = transform(hr_img).unsqueeze(0).to(device)
        
        # Dự đoán với mô hình
        with torch.no_grad():
            sr_tensor = generator(lr_tensor)
        
        # Chuyển về khoảng [0, 1] để tính PSNR
        hr_tensor = (hr_tensor + 1) / 2
        sr_tensor = (sr_tensor + 1) / 2
        
        # Tính PSNR
        psnr_value = calculate_psnr(sr_tensor, hr_tensor)
        total_psnr += psnr_value
        processed_images += 1
        
    if processed_images > 0:
        avg_psnr = total_psnr / processed_images
        print(f"Average PSNR: {avg_psnr:.2f} dB")
    else:
        print("No images were processed. Check file paths and naming conventions.")


def main():
    parser = argparse.ArgumentParser(description='Super Resolution using SISR-GAN')
    parser.add_argument('--model', type=str, required=True, help='Path to the generator model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input image path or directory')
    parser.add_argument('--output', type=str, default='results', help='Output directory or file path')
    parser.add_argument('--hr_dir', type=str, default=None, help='Directory with HR images for evaluation')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model with PSNR metric')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    
    args = parser.parse_args()
    
    # Xác định thiết bị
    device = torch.device('cuda' if torch.cuda.is_available() and args.gpu else 'cpu')
    print(f"Using device: {device}")
    
    # Tải mô hình
    generator = Generator()
    
    if os.path.exists(args.model):
        checkpoint = torch.load(args.model, map_location=device)
        
        # Kiểm tra cấu trúc checkpoint
        if isinstance(checkpoint, dict) and 'generator' in checkpoint:
            generator.load_state_dict(checkpoint['generator'])
            print(f"Loaded generator from checkpoint (epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            # Trường hợp chỉ lưu state_dict của generator
            generator.load_state_dict(checkpoint)
            print("Loaded generator state dict")
    else:
        print(f"Error: Model file {args.model} not found")
        return
    
    # Đặt mô hình ở chế độ đánh giá
    generator.to(device)
    generator.eval()
    
    # Chạy chế độ đánh giá nếu được yêu cầu
    if args.evaluate and args.hr_dir:
        evaluate(generator, args.input, args.hr_dir, device)
        return
    
    # Thực hiện super-resolution
    if os.path.isdir(args.input):
        # Xử lý cả thư mục
        inference_batch(generator, args.input, args.output, device)
    else:
        # Xử lý một ảnh đơn lẻ
        if os.path.isdir(args.output):
            output_path = os.path.join(args.output, f"sr_{os.path.basename(args.input)}")
        else:
            output_path = args.output
        inference_single_image(generator, args.input, output_path, device)


if __name__ == "__main__":
    main()