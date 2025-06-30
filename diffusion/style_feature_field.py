import torch
import torch.nn as nn
from torchvision.models import vgg19

class StyleFeatureField(nn.Module):
    def __init__(self, style_image, output_resolution=64):
        super(StyleFeatureField, self).__init__()
        self.vgg = vgg19(pretrained=True).features
        self.output_resolution = output_resolution
        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.style_image = style_image

    def forward(self):
        style_features = self.vgg(self.style_image)
        style_field = self.upsample(style_features.unsqueeze(0))
        style_field = torch.nn.functional.interpolate(style_field, size=(self.output_resolution, self.output_resolution, self.output_resolution), mode='trilinear')
        return style_field