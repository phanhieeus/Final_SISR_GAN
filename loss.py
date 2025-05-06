import torch
import torch.nn as nn
from torchvision import models


class VGGExtractor(nn.Module):
    def __init__(self, layers):
        super().__init__()
        vgg19 = models.vgg19(pretrained=True).features
        self.feature_extractor = nn.Sequential(*list(vgg19.children())[:layers])
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
    def forward(self, x):
        return self.feature_extractor(x)
    
class ContentLoss(nn.Module): #Perceptual loss dựa trên đặc trưng của VGG

    def __init__(self, vgg_extractor):
        super().__init__()
        self.vgg_extractor = vgg_extractor
        self.criterion = nn.MSELoss()
    def forward(self, sr, hr):
        sr_features = self.vgg_extractor(sr)
        hr_features = self.vgg_extractor(hr)
        loss = self.criterion(sr_features, hr_features)
        return loss

class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        # self.eta = eta = 1e-2
        self.criterion = nn.L1Loss()
    def forward(self, sr, hr):
        return self.criterion(sr, hr)
        

class AdversarialLoss(nn.Module): #Relativistic GAN loss
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
    def forward(self, D_sr, D_hr):
        g_real_loss = self.criterion((D_sr - torch.mean(D_hr)), torch.ones_like(D_sr, dtype = torch.float))
        g_fake_loss = self.criterion((D_hr - torch.mean(D_sr)), torch.zeros_like(D_sr, dtype = torch.float))
        return (g_real_loss + g_fake_loss)

class GeneratorLoss(nn.Module):
    def __init__(self, vgg_extractor):
        super().__init__()
        self.content_loss = ContentLoss(vgg_extractor)
        self.adversarial_loss = AdversarialLoss()
        self.l1_loss = L1Loss()
    def forward(self, sr, hr, D_sr, D_hr):
        content_loss = self.content_loss(sr, hr)
        l1_loss = self.l1_loss(sr, hr)
        adversarial_loss = self.adversarial_loss(D_sr, D_hr)
        return content_loss + 1e-2 * l1_loss + 5e-3 * adversarial_loss
        
class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss()
    def forward(self, D_sr, D_hr):
        relativistic_d1_loss = self.criterion((D_hr - torch.mean(D_sr)), torch.ones_like(D_sr, dtype = torch.float))
        relativistic_d2_loss = self.criterion((D_sr - torch.mean(D_hr)), torch.zeros_like(D_sr, dtype = torch.float))      

        d_loss = (relativistic_d1_loss + relativistic_d2_loss) / 2
        return d_loss