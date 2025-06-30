import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMPipeline
from significance_attention import SignificanceAttention

class HierarchicalGaussianDiffusionModule(nn.Module):
    def __init__(self, num_steps_coarse=100, num_steps_fine=900, beta_schedule='linear'):
        super(HierarchicalGaussianDiffusionModule, self).__init__()
        self.num_steps_coarse = num_steps_coarse
        self.num_steps_fine = num_steps_fine
        self.beta_schedule = beta_schedule
        self.ddpm_coarse = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to("cuda")
        self.ddpm_fine = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32").to("cuda")
        self.significance_attention = SignificanceAttention().cuda()

    def forward(self, init_gaussians_coarse, init_gaussians_fine, style_field_coarse, style_field_fine, rgb_image):
        significance_weights = self.significance_attention(rgb_image)
        significance_weights = F.interpolate(significance_weights, size=(init_gaussians_fine.shape[0], 1, 1, 1), mode='nearest')

        g_t_coarse = init_gaussians_coarse
        for t in reversed(range(self.num_steps_coarse)):
            t_batch = torch.full((g_t_coarse.shape[0],), t, dtype=torch.long).cuda()
            pred_noise_coarse = self.ddpm_coarse.predict_noise(g_t_coarse, t, style_field_coarse)
            g_t_coarse = self.ddpm_coarse.step(g_t_coarse, t, pred_noise_coarse)

        g_t_fine = init_gaussians_fine
        for t in reversed(range(self.num_steps_fine)):
            t_batch = torch.full((g_t_fine.shape[0],), t, dtype=torch.long).cuda()
            pred_noise_fine = self.ddpm_fine.predict_noise(g_t_fine, t, style_field_fine)
            adjusted_pred_noise = pred_noise_fine * significance_weights.squeeze()
            g_t_fine = self.ddpm_fine.step(g_t_fine, t, adjusted_pred_noise)

        combined_gaussians = torch.cat((g_t_coarse, g_t_fine), dim=0)
        return combined_gaussians