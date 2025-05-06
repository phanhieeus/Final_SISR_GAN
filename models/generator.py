import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return F.relu(out)

class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super(AttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.conv2 = nn.Conv2d(channels // 8, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = F.relu(self.conv1(x))
        attention = self.sigmoid(self.conv2(attention))
        return x * attention

class Generator(nn.Module):
    def __init__(self, num_residual_blocks=16, num_filters=64, use_attention=True):
        super(Generator, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=9, padding=4)
        self.relu = nn.ReLU(inplace=True)
        
        # Residual blocks
        res_blocks = []
        for _ in range(num_residual_blocks):
            res_blocks.append(ResidualBlock(num_filters))
        self.res_blocks = nn.Sequential(*res_blocks)
        
        # Attention block
        self.use_attention = use_attention
        if use_attention:
            self.attention = AttentionBlock(num_filters)
        
        # Upsampling blocks
        self.upsampling = nn.Sequential(
            nn.Conv2d(num_filters, num_filters * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters * 4, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution
        self.conv2 = nn.Conv2d(num_filters, 3, kernel_size=9, padding=4)
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.conv1(x))
        residual = out
        
        out = self.res_blocks(out)
        if self.use_attention:
            out = self.attention(out)
        
        out = out + residual
        out = self.upsampling(out)
        out = self.tanh(self.conv2(out))
        
        return out 