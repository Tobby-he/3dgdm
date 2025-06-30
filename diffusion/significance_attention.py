import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19

class SignificanceAttention(nn.Module):
    def __init__(self, pretrained=True):
        super(SignificanceAttention, self).__init__()
        self.vgg = vgg19(pretrained=pretrained).features
        self.fuse = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 1, kernel_size=1)
        )

    def forward(self, x):
        features = []
        for i in range(len(self.vgg)):
            x = self.vgg[i](x)
            if i in {1, 3, 6, 8, 11, 13, 15, 17, 20, 22, 24, 26}:
                features.append(x)

        fused_features = torch.cat([
            F.interpolate(features[-1], size=features[-5].shape[2:], mode='bilinear', align_corners=True),
            features[-5]
        ], dim=1)
        significance_map = self.fuse(fused_features)
        significance_map = torch.sigmoid(significance_map)
        return significance_map