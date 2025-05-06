import torch.nn as nn


# LR: 64x64
# HR: 256x256

class discriminator_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.LeakyReLU = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.LeakyReLU(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc_conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            discriminator_block(64, 64, kernel_size=3, stride=2, padding=1),
            discriminator_block(64, 128, kernel_size=3, stride=1, padding=1),
            discriminator_block(128, 128, kernel_size=3, stride=2, padding=1),
            discriminator_block(128, 256, kernel_size=3, stride=1, padding=1),
            discriminator_block(256, 256, kernel_size=3, stride=2, padding=1),
            discriminator_block(256, 512, kernel_size=3, stride=1, padding=1),
            discriminator_block(512, 512, kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(1024, 1))
        # self.fc = nn.Linear(512 * 16 * 16, 1)
        

    def forward(self, x):
        x = self.disc_conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x