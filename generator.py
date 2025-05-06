import torch
import torch.nn as nn


class ResidualDenseBlock(nn.Module):
    def __init__(self, in_channel=64, inc_channel=32, beta=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, inc_channel, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channel + inc_channel, inc_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channel + 2 * inc_channel, inc_channel, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channel + 3 * inc_channel, inc_channel, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channel + 4 * inc_channel, in_channel, kernel_size=3, stride=1, padding=1)
        self.LeakyReLU = nn.LeakyReLU()
        self.beta = beta

    def forward(self, x):
        block1 = self.LeakyReLU(self.conv1(x)) # x_c=64, block1_c=32
        block2 = self.LeakyReLU(self.conv2(torch.cat((block1, x), dim=1))) # block2_c = 32
        block3 = self.LeakyReLU(self.conv3(torch.cat((block2, block1, x), dim=1))) #block3_c =32
        block4 = self.LeakyReLU(self.conv4(torch.cat((block3, block2, block1, x), dim=1)))
        out = self.conv5(torch.cat((block4, block3, block2, block1, x), dim=1))
        return x + self.beta * out #channel = 64

class ResidualInResidualDenseBlock(nn.Module):
    def __init__(self, in_channel=64, out_channel=32, beta=0.2):
        super().__init__()
        self.RDB = ResidualDenseBlock(in_channel, out_channel)
        self.beta = beta

    def forward(self, x):
        RDB1 = self.RDB(x) # x Channel = 64
        RDB2 = self.RDB(RDB1)
        RDB3 = self.RDB(RDB2)
        return x + self.beta * RDB3 # channel = 64

class Generator(nn.Module):
    def __init__(self, BLOCK=23, scale=2):
        
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.res_blocks = nn.Sequential(*[ResidualInResidualDenseBlock() for _ in range(BLOCK)])
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        res = self.res_blocks(x)
        res = self.conv2(res)
        x = x + res
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        x = self.conv3(x)
        x = self.pixel_shuffle(x)
        x = self.conv4(x)              
        return x