import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralNorm:
    def __init__(self, module, name='weight', power_iterations=1):
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self._make_params()

    def _make_params(self):
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1))
        v = nn.Parameter(w.data.new(width).normal_(0, 1))
        u.requires_grad = False
        v.requires_grad = False
        self.module.register_buffer('u', u)
        self.module.register_buffer('v', v)

    def _power_method(self, w, eps=1e-12):
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v = F.normalize(torch.mv(w.view(height, -1).t(), self.module.u), dim=0, eps=eps)
            u = F.normalize(torch.mv(w.view(height, -1), v), dim=0, eps=eps)
        sigma = u.dot(w.view(height, -1).mv(v))
        return u, v, sigma

    def forward(self):
        w = getattr(self.module, self.name)
        u, v, sigma = self._power_method(w)
        self.module.u.copy_(u)
        self.module.v.copy_(v)
        w_sn = w / sigma
        setattr(self.module, self.name, w_sn)

class Discriminator(nn.Module):
    def __init__(self, num_filters=64, num_layers=3, use_spectral_norm=True):
        super(Discriminator, self).__init__()
        
        def conv_block(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
            if use_spectral_norm:
                conv = SpectralNorm(conv)
            return conv

        layers = []
        in_channels = 3
        
        # First layer
        layers.append(conv_block(in_channels, num_filters, kernel_size=3, stride=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Middle layers
        for i in range(num_layers):
            in_channels = num_filters * (2 ** i)
            out_channels = num_filters * (2 ** (i + 1))
            layers.append(conv_block(in_channels, out_channels, stride=2))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            layers.append(conv_block(out_channels, out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final layer
        final_channels = num_filters * (2 ** num_layers)
        layers.append(conv_block(final_channels, final_channels, stride=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(conv_block(final_channels, 1, kernel_size=1, stride=1, padding=0))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x) 