import torch
from torch.utils.tensorboard import SummaryWriter
import os
import uuid
import numpy as np
from argparse import Namespace
from scene import Scene, GaussianModel
from render import render
from utils.loss_utils import cal_adain_style_loss, cal_mse_content_loss
from diffusion.significance_attention import SignificanceAttention
from diffusion.hierarchical_gaussian_diffusion import HierarchicalGaussianDiffusionModule
from diffusion.style_feature_field import StyleFeatureField
import torchvision
from PIL import Image
import torchvision.transforms as T
from torchvision.models import vgg19  # 导入vgg19模型
import torch.nn.functional as F  # 导入F模块

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = SummaryWriter(args.model_path)

    return tb_writer

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    style_image_path = "path/to/your/style/image.jpg"
    style_image = torchvision.transforms.Compose([
        torchvision.transforms.Resize(size=(256, 256)),
        torchvision.transforms.ToTensor(),
    ])(Image.open(style_image_path)).cuda()[None, :3, :, :]

    style_feature_field_coarse = StyleFeatureField(style_image, output_resolution=32).cuda()
    style_feature_field_fine = StyleFeatureField(style_image, output_resolution=128).cuda()

    hgdm = HierarchicalGaussianDiffusionModule(num_steps_coarse=100, num_steps_fine=900, beta_schedule='linear').cuda()

    significance_attention = SignificanceAttention().cuda()

    scene = Scene(dataset, GaussianModel(dataset.sh_degree))
    gaussians = scene.gaussians

    first_iter = 0
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    init_gaussians_coarse = gaussians.get_xyz.clone()
    init_gaussians_fine = gaussians.get_xyz.clone()

    optimizer = torch.optim.Adam(list(gaussians.parameters()), lr=opt.lr)

    viewpoint_cameras = dataset.getTrainCameras()
    viewpoint_cam = viewpoint_cameras[0] if viewpoint_cameras else None
    background = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device="cuda")

    if viewpoint_cam is None:
        raise ValueError("No cameras found in the dataset.")

    vgg_model = vgg19(pretrained=True).features.cuda()  # 定义vgg模型
    style_image_normalized = torchvision.transforms.functional.normalize(style_image, mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225])
    style_features = vgg_model(style_image_normalized)  # 计算风格特征

    for iteration in range(first_iter, opt.iterations + 1):
        rgb_image = render(viewpoint_cam, gaussians, pipe, background)["render"]

        if iteration < opt.iterations * 0.3:
            style_field_coarse = style_feature_field_coarse()
            with torch.no_grad():
                combined_gaussians = hgdm(init_gaussians_coarse, init_gaussians_fine, style_field_coarse, None, rgb_image)
            gaussians.restore(combined_gaussians[:init_gaussians_coarse.shape[0]])
        else:
            style_field_fine = style_feature_field_fine()
            with torch.no_grad():
                combined_gaussians = hgdm(init_gaussians_coarse, init_gaussians_fine, None, style_field_fine, rgb_image)
            gaussians.restore(combined_gaussians[init_gaussians_coarse.shape[0]:])

        render_pkg = render(viewpoint_cam, gaussians, pipe, background)
        current_image = render_pkg["render"]

        gt_image = render_pkg["gt_image"].unsqueeze(0)
        rendered_rgb = render_pkg["render"].unsqueeze(0)

        gt_image_features = vgg_model(gt_image)
        rendered_rgb_features = vgg_model(rendered_rgb)
        content_loss = cal_mse_content_loss(gt_image_features[-1], rendered_rgb_features[-1])
        style_loss = cal_adain_style_loss(style_features, rendered_rgb_features[0])
        loss = content_loss + style_loss * opt.style_weight

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % opt.log_interval == 0:
            print(f"Iteration {iteration}, Loss: {loss.item()}, Content Loss: {content_loss.item()}, Style Loss: {style_loss.item()}")

        if iteration in saving_iterations:
            os.makedirs(opt.output_dir, exist_ok=True)
            torch.save(gaussians.state_dict(), os.path.join(opt.output_dir, f"model_iter_{iteration}.pth"))

    print("Training completed.")

if __name__ == "__main__":
    class Args:
        def __init__(self):
            self.model_path = "./output/train/artistic/default"
            self.data_device = "cuda"
            self.lr = 0.001
            self.iterations = 10000
            self.log_interval = 10
            self.style_weight = 10.0
            self.content_preserve = False
            self.wikiartdir = "./datasets/wikiart"

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
    testing_iterations = [1000, 2000, 3000]
    saving_iterations = [500, 1000, 1500]
    checkpoint_iterations = [250, 500, 750]
    checkpoint = None
    debug_from = 0

    training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from)