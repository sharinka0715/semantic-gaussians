import os
import torch
import torchvision
from tqdm import tqdm
from omegaconf import OmegaConf

from model import GaussianModel, render
from model.openseg_predictor import OpenSeg
from scene import Scene
from utils.system_utils import searchForMaxIteration, set_seed
from utils.sh_utils import SH2RGB
from dataset.scannet.scannet_constants import SCANNET_COLOR_MAP_20


def render_sets(config):
    with torch.no_grad():
        if config.model.dynamic:
            config.scene.scene_path = os.path.join(config.scene.scene_path, str(config.model.dynamic_t))
        scene = Scene(config.scene)
        gaussians = GaussianModel(config.model.sh_degree)

        if config.model.model_dir:
            if config.model.dynamic:
                loaded_iter = config.model.dynamic_t
                gaussians.load_dynamic_npz(os.path.join(config.model.model_dir, "params.npz"), config.model.dynamic_t)
            else:
                if config.model.load_iteration == -1:
                    loaded_iter = searchForMaxIteration(os.path.join(config.model.model_dir, "point_cloud"))
                else:
                    loaded_iter = config.model.load_iteration
                print("Loading trained model at iteration {}".format(loaded_iter))
                gaussians.load_ply(
                    os.path.join(
                        config.model.model_dir,
                        "point_cloud",
                        f"iteration_{loaded_iter}",
                        "point_cloud.ply",
                    )
                )
        else:
            raise NotImplementedError

        bg_color = [1, 1, 1] if config.scene.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # render train
        views = scene.getTrainCameras()
        render_path = os.path.join(config.model.model_dir, "train", "ours_{}".format(loaded_iter), "renders")
        gts_path = os.path.join(config.model.model_dir, "train", "ours_{}".format(loaded_iter), "gt")

        os.makedirs(render_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            rendering = render(view, gaussians, config.pipeline, background)["render"]
            gt = view.original_image[0:3, :, :]
            torchvision.utils.save_image(rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png"))


def get_text_features(model_2d, labelset):
    colormap = [SCANNET_COLOR_MAP_20[k] for k in SCANNET_COLOR_MAP_20]
    labelset = ["other", *labelset]

    palette = torch.tensor(colormap).reshape(-1).cuda()
    text_features = model_2d.extract_text_feature(labelset).float()

    return palette, text_features


def render_palette(label, palette):
    shape = label.shape
    label = label.reshape(-1)
    new_3d = torch.zeros((label.shape[0], 3)).cuda()
    u_index = torch.unique(label)
    for index in u_index:
        new_3d[label == index] = torch.tensor(
            [
                palette[index * 3] / 255.0,
                palette[index * 3 + 1] / 255.0,
                palette[index * 3 + 2] / 255.0,
            ]
        ).cuda()

    return new_3d.reshape(*shape, 3).permute(2, 0, 1)


def render_sem(config):
    with torch.no_grad():
        if config.model.dynamic:
            config.scene.scene_path = os.path.join(config.scene.scene_path, str(config.model.dynamic_t))
        scene = Scene(config.scene)
        gaussians = GaussianModel(config.model.sh_degree)
        model_2d = OpenSeg("./weights/openseg_exported_clip", "ViT-L/14@336px")
        palette, text_features = get_text_features(
            model_2d,
            [
                "wall",
                "floor",
                "window",
                "blanket",
                "table",
                "chair",
                "sofa",
                "television",
                "door",
                "piano",
                "plants",
                "bookshelf",
            ],
        )

        if config.model.model_dir:
            if config.model.dynamic:
                loaded_iter = config.model.dynamic_t
                gaussians.load_dynamic_npz(os.path.join(config.model.model_dir, "params.npz"), config.model.dynamic_t)
            else:
                if config.model.load_iteration == -1:
                    loaded_iter = searchForMaxIteration(os.path.join(config.model.model_dir, "point_cloud"))
                else:
                    loaded_iter = config.model.load_iteration
                print("Loading trained model at iteration {}".format(loaded_iter))
                gaussians.load_ply(
                    os.path.join(
                        config.model.model_dir,
                        "point_cloud",
                        f"iteration_{loaded_iter}",
                        "point_cloud.ply",
                    )
                )
        else:
            raise NotImplementedError

        bg_color = [1, 1, 1] if config.scene.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # render train
        views = scene.getTrainCameras()
        render_path = os.path.join(config.model.model_dir, "train", "ours_{}".format(loaded_iter), "renders")
        gts_path = os.path.join(config.model.model_dir, "train", "ours_{}".format(loaded_iter), "gt")
        sem_path = os.path.join(config.model.model_dir, "train", "ours_{}".format(loaded_iter), "sem")

        os.makedirs(render_path, exist_ok=True)
        os.makedirs(gts_path, exist_ok=True)
        os.makedirs(sem_path, exist_ok=True)

        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            rendering = render(view, gaussians, config.pipeline, background)["render"]
            gt = view.original_image[0:3, :, :]
            gt_path = str(view.image_path)
            features = model_2d.extract_image_feature(
                gt_path,
                [gt.shape[1], gt.shape[2]],
            ).float()
            sim = torch.einsum("cq,qhw->chw", text_features.cpu(), features)
            label = sim.argmax(dim=0)
            sem = render_palette(label, palette)
            torchvision.utils.save_image(rendering, os.path.join(render_path, "{0:05d}".format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, "{0:05d}".format(idx) + ".png"))
            torchvision.utils.save_image(sem, os.path.join(sem_path, "{0:05d}".format(idx) + ".png"))


if __name__ == "__main__":
    config = OmegaConf.load("./config/fusion_lerf.yaml")
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)
    print(OmegaConf.to_yaml(config))

    set_seed(config.pipeline.seed)
    render_sem(config)
