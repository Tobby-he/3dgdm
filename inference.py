import torch
from diffusion.hierarchical_gaussian_diffusion import HierarchicalGaussianDiffusionModule
from diffusion.style_feature_field import StyleFeatureField
from diffusion.significance_attention import SignificanceAttention
from scene.gaussian_model import GaussianModel
from scene import Scene
import torchvision.transforms as T
from torchvision.models import vgg19  # 导入vgg19模型
from PIL import Image
import torchvision
import torch.nn.functional as F  # 导入F模块
import numpy as np  # 导入numpy模块
from render import render  # 导入render函数

def inference(dataset, opt, pipe, checkpoint_path, style_image_path):
    gaussians = GaussianModel(dataset.sh_degree)
    checkpoint = torch.load(checkpoint_path)
    gaussians.restore(checkpoint["gaussians"], opt)

    style_image = Image.open(style_image_path).convert('RGB')
    style_image = T.Compose([
        T.Resize(size=(256, 256)),
        T.ToTensor(),
    ])(style_image).cuda()[None, :3, :, :]

    style_feature_field_coarse = StyleFeatureField(style_image, output_resolution=32).cuda()
    style_feature_field_fine = StyleFeatureField(style_image, output_resolution=128).cuda()

    hgdm = HierarchicalGaussianDiffusionModule(num_steps_coarse=100, num_steps_fine=900, beta_schedule='linear').cuda()

    init_gaussians_coarse = gaussians.get_xyz.clone()
    init_gaussians_fine = gaussians.get_xyz.clone()

    significance_attention = SignificanceAttention().cuda()

    rgb_image = torch.randn((1, 3, 256, 256)).cuda()  # 随机生成的RGB图像

    vgg_model = vgg19(pretrained=True).features.cuda()  # 定义vgg模型
    style_image_normalized = torchvision.transforms.functional.normalize(style_image, mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])
    style_features = vgg_model(style_image_normalized)  # 计算风格特征

    with torch.no_grad():
        significance_weights = significance_attention(rgb_image)
        significance_weights = F.interpolate(significance_weights, size=(init_gaussians_fine.shape[0], 1, 1, 1), mode='nearest')

        style_field_coarse = style_feature_field_coarse()
        combined_gaussians_coarse = hgdm(init_gaussians_coarse, init_gaussians_fine, style_field_coarse, None, rgb_image)

        style_field_fine = style_feature_field_fine()
        combined_gaussians_fine = hgdm(init_gaussians_coarse, init_gaussians_fine, None, style_field_fine, rgb_image)

        final_gaussians = torch.cat((combined_gaussians_coarse[:init_gaussians_coarse.shape[0]], combined_gaussians_fine[init_gaussians_coarse.shape[0]:]), dim=0)

    gaussians.restore(final_gaussians)
    viewpoint_cam = dataset.getTrainCameras()[0]  # 获取训练相机
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")
    render_pkg = render(viewpoint_cam, gaussians, pipe, background)
    final_image = render_pkg["render"]
    torchvision.utils.save_image(final_image, "stylized_image.png")

if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.model_path = "./output/train/artistic/default"
            self.data_device = "cuda"

    args = Args()

    class Dataset:
        def __init__(self):
            self.sh_degree = 3

            class Camera:
                def __init__(self):
                    self.uid = 0
                    self.R = np.eye(3)
                    self.T = np.zeros(3)
                    self.FovX = 60.0
                    self.FovY = 60.0
                    self.image = torch.zeros((3, 256, 256))
                    self.image_name = "camera_0"

            self.getTrainCameras = lambda: [Dataset.Camera()]

    dataset = Dataset()
    opt = args
    pipe = None
    checkpoint_path = "./output/Barn/artistic/default/chkpnt/feature.pth"
    style_image_path = "./datasets/wikiart/style_image.jpg"

    inference(dataset, opt, pipe, checkpoint_path, style_image_path)