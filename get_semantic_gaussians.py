import os
import imageio

import torch
import torchvision
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from omegaconf import OmegaConf
import skimage.transform as sktf

from model import GaussianModel
from scene import Scene
from utils.system_utils import set_seed, searchForMaxIteration

from model.mink_unet import mink_unet
from model.openseg_predictor import OpenSeg
from dataset.fusion_utils import Voxelizer

import MinkowskiEngine as ME


def init_dir(config):
    print("Distill folder: {}".format(config.distill.model_dir))
    weights_dir = os.path.join(config.distill.model_dir, "weights")
    weights = os.listdir(weights_dir)
    weights.sort(key=lambda x: int(x))
    iteration = weights[-1]
    ckpt_path = os.path.join(weights_dir, str(iteration), "model.pth")
    return ckpt_path


def evaluate(config):
    model_3d = mink_unet(in_channels=52, out_channels=768 * 3, D=3, arch=config.distill.model_3d).cuda()
    ckpt_path = init_dir(config)
    model_3d.load_state_dict(torch.load(ckpt_path))

    with torch.no_grad():
        eval_config = deepcopy(config)
        gaussians = GaussianModel(eval_config.model.sh_degree)
        if config.model.load_iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(config.model.model_dir, "point_cloud"))
        else:
            loaded_iter = config.model.load_iteration
        gaussians.load_ply(
            os.path.join(
                eval_config.model.model_dir,
                "point_cloud",
                f"iteration_{loaded_iter}",
                "point_cloud.ply",
            )
        )
        gaussians.create_semantic(768)

        locs, features = gaussians.get_locs_and_features(False)
        voxelizer = Voxelizer(voxel_size=config.distill.voxel_size)
        locs, features, _, _, vox_ind = voxelizer.voxelize(locs, features, None, return_ind=True)
        locs = torch.from_numpy(locs).int()
        locs = torch.cat([torch.ones(locs.shape[0], 1, dtype=torch.int), locs], dim=1)
        features = torch.from_numpy(features).float()
        vox_ind = torch.from_numpy(vox_ind).cuda()

        sinput = ME.SparseTensor(features.cuda(), locs.cuda())
        z = 0
        output = model_3d(sinput).F[:, z * 768 : (z + 1) * 768]
        output /= output.norm(dim=-1, keepdim=True) + 1e-8
        gaussians._features_semantic[vox_ind] = output

        mask_entire = torch.ones(gaussians._xyz.shape[0], dtype=torch.bool)

        torch.save(
            {
                "feat": gaussians._features_semantic.cpu().half()[mask_entire],
                "mask_full": mask_entire,
            },
            os.path.join(config.fusion.out_dir + "/output.pt"),
        )


if __name__ == "__main__":
    config = OmegaConf.load("./config/fusion_scannet.yaml")
    print(OmegaConf.to_yaml(config))

    set_seed(config.pipeline.seed)

    evaluate(config)
