import os
import imageio

import torch
import torchvision
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from omegaconf import OmegaConf
import skimage.transform as sktf

from model import GaussianModel, render_sem, render_chn
from scene import Scene
from utils.system_utils import set_seed

from model.mink_unet import mink_unet
from model.openseg_predictor import OpenSeg
from dataset.fusion_utils import Voxelizer
from dataset.scannet.scannet_constants import (
    SCANNET_CLASS_LABELS_20,
    SCANNET_VALID_CLASS_IDS_20,
    SCANNET_COLOR_MAP_20,
    SCANNET_CLASS_LABELS_200,
    SCANNET_COLOR_MAP_200,
    SCANNET_VALID_CLASS_IDS_200,
)
from dataset.scannet.label_mapping import read_label_mapping

from utils import metric

import MinkowskiEngine as ME


def init_dir(config):
    print("Distill folder: {}".format(config.distill.model_dir))
    weights_dir = os.path.join(config.distill.model_dir, "weights")
    if config.distill.iteration == -1:
        weights = os.listdir(weights_dir)
        weights.sort(key=lambda x: int(x))
        iteration = weights[-1]
    else:
        iteration = config.distill.iteration
    ckpt_path = os.path.join(weights_dir, str(iteration), "model.pth")
    return ckpt_path


def get_text_features(model_2d, dataset_name="scannet_20"):
    if dataset_name == "scannet_20":
        labelset = list(SCANNET_CLASS_LABELS_20)
        valid_classes = SCANNET_VALID_CLASS_IDS_20
        color_map = SCANNET_COLOR_MAP_20
        labelset[-1] = "other"  # for OpenSeg
    elif dataset_name == "scannet_200":
        labelset = list(SCANNET_CLASS_LABELS_200)
        valid_classes = SCANNET_VALID_CLASS_IDS_200
        color_map = SCANNET_COLOR_MAP_200

    scannet_palette = []
    mapping = []
    for k in valid_classes:
        scannet_palette.append(torch.tensor(color_map[k]))
        mapping.append(k)
    # add unlabeled label and palette
    labelset.append("unlabeled")
    scannet_palette.append(torch.tensor(color_map[0]))
    mapping.append(0)

    palette = torch.cat(scannet_palette).cuda()
    text_features = model_2d.extract_text_feature(labelset).float()

    return palette, text_features, mapping


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


def get_mapped_label(config, image_path, label_mapping):
    # label_path = image_path.replace("./train", "./label-filt").replace(".jpg", ".png")
    label_path = str(image_path).replace("color", "label-filt").replace(".jpg", ".png")
    if not os.path.exists(label_path):
        return None

    label_img = np.array(imageio.imread(label_path))
    label_img = sktf.resize(label_img, [config.eval.height, config.eval.width], order=0, preserve_range=True)
    mapped = np.copy(label_img)
    for k, v in label_mapping.items():
        mapped[label_img == k] = v
    label_img = mapped.astype(np.uint8)

    return label_img


def evaluate(config):
    if config.scene.dataset_name == "scannet_20":
        config.scene.num_classes = 20
        label_mapping = read_label_mapping("./dataset/scannet/scannetv2-labels.combined.tsv")
    elif config.scene.dataset_name == "scannet_200":
        config.scene.num_classes = 200
        label_mapping = read_label_mapping("./dataset/scannet/scannetv2-labels.combined.tsv", label_to="id")

    if config.distill.feature_rotation:
        model_3d = mink_unet(in_channels=56, out_channels=768, D=3, arch=config.distill.model_3d).cuda()
    else:
        model_3d = mink_unet(in_channels=52, out_channels=768, D=3, arch=config.distill.model_3d).cuda()

    ckpt_path = init_dir(config)
    model_3d.load_state_dict(torch.load(ckpt_path))

    eval_2d_and_fusion(config, model_3d, label_mapping)


def eval_2d(config, model_3d, label_mapping):
    eval_scene = os.listdir(config.model.model_dir)
    eval_scene.sort()

    model_2d = OpenSeg(None, "ViT-L/14@336px")

    bg_color = [1] * 768 if config.scene.white_background else [0] * 768
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    confusion = np.zeros((config.scene.num_classes + 1, config.scene.num_classes), dtype=np.ulonglong)

    for i, scene_name in enumerate(tqdm(eval_scene)):
        if i > 20:
            break
        torch.cuda.empty_cache()
        with torch.no_grad():
            eval_config = deepcopy(config)
            eval_config.scene.scene_path = os.path.join(eval_config.scene.scene_path, scene_name)
            eval_config.model.model_dir = os.path.join(eval_config.model.model_dir, scene_name)
            scene = Scene(eval_config.scene)
            gaussians = GaussianModel(eval_config.model.sh_degree)
            gaussians.load_ply(
                os.path.join(
                    eval_config.model.model_dir,
                    "point_cloud",
                    "iteration_10000",
                    "point_cloud.ply",
                )
            )
            gaussians.create_semantic(768)

            locs, features = gaussians.get_locs_and_features(eval_config.distill.feature_rotation)
            voxelizer = Voxelizer(voxel_size=config.distill.voxel_size)
            locs, features, _, _, vox_ind = voxelizer.voxelize(locs, features, None, return_ind=True)
            locs = torch.from_numpy(locs).int()
            locs = torch.cat([torch.ones(locs.shape[0], 1, dtype=torch.int), locs], dim=1)
            features = torch.from_numpy(features).float()
            vox_ind = torch.from_numpy(vox_ind).cuda()

            sinput = ME.SparseTensor(features.cuda(), locs.cuda())
            output = model_3d(sinput).F[:, 768 * 0 : 768 * 1]
            output /= output.norm(dim=-1, keepdim=True) + 1e-8

            views = scene.getTrainCameras()
            # out_path = "eval_samples/"
            # os.makedirs(out_path, exist_ok=True)
            gaussians._features_semantic[vox_ind] = output
            palette, text_features, mapping = get_text_features(model_2d, dataset_name=config.scene.dataset_name)
            for idx, view in enumerate(views):
                if idx % 5 != 0:
                    continue
                view.cuda()
                gt_path = str(view.image_path)
                label_img = get_mapped_label(config, gt_path, label_mapping)
                if label_img is None:
                    continue
                mapped = np.ones_like(label_img) * config.scene.num_classes
                for i in range(len(mapping)):
                    mapped[label_img == mapping[i]] = i
                label_img = torch.from_numpy(mapped).int().cpu()

                if config.eval.pred_on_3d:
                    sim = torch.einsum("cq,dq->dc", text_features, gaussians._features_semantic)
                    label_soft = sim.softmax(dim=1)
                    label_hard = torch.nn.functional.one_hot(sim.argmax(dim=1), num_classes=label_soft.shape[1]).float()
                    rendering = render_chn(
                        view,
                        gaussians,
                        eval_config.pipeline,
                        background,
                        num_channels=label_soft.shape[1],
                        override_color=label_soft,
                        override_shape=[config.eval.width, config.eval.height],
                    )["render"]
                    label = rendering.argmax(dim=0).cpu()
                else:
                    rendering = render_sem(
                        view,
                        gaussians,
                        eval_config.pipeline,
                        background,
                        override_color=gaussians._features_semantic,
                        override_shape=[config.eval.width, config.eval.height],
                    )["render"]
                    rendering = rendering / (rendering.norm(dim=0, keepdim=True) + 1e-8)
                    sim = torch.einsum("cq,qhw->chw", text_features, rendering)
                    label = sim.argmax(dim=0).cpu()

                # sem = render_palette(label, palette)
                # sem_gt = render_palette(label_img, palette)
                # torchvision.utils.save_image(sem, os.path.join(out_path, "{0:05d}".format(idx) + "_render.png"))
                # torchvision.utils.save_image(sem_gt, os.path.join(out_path, "{0:05d}".format(idx) + ".png"))
                confusion += metric.confusion_matrix(
                    label.cpu().numpy().reshape(-1), label_img.cpu().numpy().reshape(-1), config.scene.num_classes
                )

    metric.evaluate_confusion(confusion, stdout=True, dataset=config.scene.dataset_name)


def eval_fusion(config, model_3d, label_mapping):
    eval_scene = os.listdir(config.model.model_dir)
    eval_scene.sort()

    model_2d = OpenSeg(None, "ViT-L/14@336px")

    bg_color = [1] * 768 if config.scene.white_background else [0] * 768
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    confusion = np.zeros((config.scene.num_classes + 1, config.scene.num_classes), dtype=np.ulonglong)

    for i, scene_name in enumerate(tqdm(eval_scene)):
        if i > 20:
            break
        torch.cuda.empty_cache()
        with torch.no_grad():
            eval_config = deepcopy(config)
            eval_config.scene.scene_path = os.path.join(eval_config.scene.scene_path, scene_name)
            eval_config.model.model_dir = os.path.join(eval_config.model.model_dir, scene_name)
            scene = Scene(eval_config.scene)
            gaussians = GaussianModel(eval_config.model.sh_degree)
            gaussians.load_ply(
                os.path.join(
                    eval_config.model.model_dir,
                    "point_cloud",
                    "iteration_10000",
                    "point_cloud.ply",
                )
            )
            gaussians.create_semantic(768)

            feature_path = os.path.join(eval_config.fusion.out_dir, scene_name, "0.pt")
            gt = torch.load(feature_path)
            feat, mask_full = gt["feat"], gt["mask_full"]

            views = scene.getTrainCameras()
            # out_path = "eval_samples/"
            # os.makedirs(out_path, exist_ok=True)
            gaussians._features_semantic[mask_full] = feat.float().cuda()
            palette, text_features, mapping = get_text_features(model_2d, dataset_name=config.scene.dataset_name)
            for idx, view in enumerate(views):
                if idx % 5 != 0:
                    continue
                view.cuda()
                gt_path = str(view.image_path)
                label_img = get_mapped_label(config, gt_path, label_mapping)
                if label_img is None:
                    continue
                mapped = np.ones_like(label_img) * config.scene.num_classes
                for i in range(len(mapping)):
                    mapped[label_img == mapping[i]] = i
                label_img = torch.from_numpy(mapped).int().cpu()

                if config.eval.pred_on_3d:
                    sim = torch.einsum("cq,dq->dc", text_features, gaussians._features_semantic)
                    label_soft = sim.softmax(dim=1)
                    label_hard = torch.nn.functional.one_hot(sim.argmax(dim=1), num_classes=label_soft.shape[1]).float()
                    rendering = render_chn(
                        view,
                        gaussians,
                        eval_config.pipeline,
                        background,
                        num_channels=label_soft.shape[1],
                        override_color=label_soft,
                        override_shape=[config.eval.width, config.eval.height],
                    )["render"]
                    label = rendering.argmax(dim=0).cpu()
                else:
                    rendering = render_sem(
                        view,
                        gaussians,
                        eval_config.pipeline,
                        background,
                        override_color=gaussians._features_semantic,
                        override_shape=[config.eval.width, config.eval.height],
                    )["render"]
                    rendering = rendering / (rendering.norm(dim=0, keepdim=True) + 1e-8)
                    sim = torch.einsum("cq,qhw->chw", text_features, rendering)
                    label = sim.argmax(dim=0).cpu()

                # sem = render_palette(label, palette)
                # sem_gt = render_palette(label_img, palette)
                # torchvision.utils.save_image(sem, os.path.join(out_path, "{0:05d}".format(idx) + "_render.png"))
                # torchvision.utils.save_image(sem_gt, os.path.join(out_path, "{0:05d}".format(idx) + ".png"))
                confusion += metric.confusion_matrix(
                    label.cpu().numpy().reshape(-1), label_img.cpu().numpy().reshape(-1), config.scene.num_classes
                )

    metric.evaluate_confusion(confusion, stdout=True, dataset=config.scene.dataset_name)


def eval_2d_and_fusion(config, model_3d, label_mapping):
    eval_scene = os.listdir(config.model.model_dir)
    eval_scene.sort()

    model_2d = OpenSeg(None, "ViT-L/14@336px")

    bg_color = [1] * 768 if config.scene.white_background else [0] * 768
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    confusion = np.zeros((config.scene.num_classes + 1, config.scene.num_classes), dtype=np.ulonglong)

    for i, scene_name in enumerate(tqdm(eval_scene)):
        if i > 20:
            break
        torch.cuda.empty_cache()
        with torch.no_grad():
            eval_config = deepcopy(config)
            eval_config.scene.scene_path = os.path.join(eval_config.scene.scene_path, scene_name)
            eval_config.model.model_dir = os.path.join(eval_config.model.model_dir, scene_name)
            scene = Scene(eval_config.scene)
            gaussians = GaussianModel(eval_config.model.sh_degree)
            gaussians.load_ply(
                os.path.join(
                    eval_config.model.model_dir,
                    "point_cloud",
                    "iteration_10000",
                    "point_cloud.ply",
                )
            )
            gaussians.create_semantic(768 * 2)

            locs, features = gaussians.get_locs_and_features(eval_config.distill.feature_rotation)
            voxelizer = Voxelizer(voxel_size=config.distill.voxel_size)
            locs, features, _, _, vox_ind = voxelizer.voxelize(locs, features, None, return_ind=True)
            locs = torch.from_numpy(locs).int()
            locs = torch.cat([torch.ones(locs.shape[0], 1, dtype=torch.int), locs], dim=1)
            features = torch.from_numpy(features).float()
            vox_ind = torch.from_numpy(vox_ind).cuda()

            sinput = ME.SparseTensor(features.cuda(), locs.cuda())
            output = model_3d(sinput).F[:, 768 * 0 : 768 * 1]
            output /= output.norm(dim=-1, keepdim=True) + 1e-8

            feature_path = os.path.join(eval_config.fusion.out_dir, scene_name, "0.pt")
            gt = torch.load(feature_path)
            feat, mask_full = gt["feat"], gt["mask_full"]

            views = scene.getTrainCameras()
            # out_path = "eval_samples/"
            # os.makedirs(out_path, exist_ok=True)
            gaussians._features_semantic[mask_full, :768] = feat.float().cuda()
            gaussians._features_semantic[vox_ind, 768:] = output
            palette, text_features, mapping = get_text_features(model_2d, dataset_name=config.scene.dataset_name)
            text_features = torch.cat([text_features, text_features], dim=1)
            for idx, view in enumerate(views):
                if idx % 5 != 0:
                    continue
                view.cuda()
                gt_path = str(view.image_path)
                label_img = get_mapped_label(config, gt_path, label_mapping)
                if label_img is None:
                    continue
                mapped = np.ones_like(label_img) * config.scene.num_classes
                for i in range(len(mapping)):
                    mapped[label_img == mapping[i]] = i
                label_img = torch.from_numpy(mapped).int().cpu()

                if config.eval.pred_on_3d:
                    sim = torch.einsum("cq,dq->dc", text_features, gaussians._features_semantic)
                    label_soft = sim.softmax(dim=1)
                    label_hard = torch.nn.functional.one_hot(sim.argmax(dim=1), num_classes=label_soft.shape[1]).float()
                    rendering = render_chn(
                        view,
                        gaussians,
                        eval_config.pipeline,
                        background,
                        num_channels=label_soft.shape[1],
                        override_color=label_soft,
                        override_shape=[config.eval.width, config.eval.height],
                    )["render"]
                    label = rendering.argmax(dim=0).cpu()
                else:
                    rendering = render_sem(
                        view,
                        gaussians,
                        eval_config.pipeline,
                        background,
                        override_color=gaussians._features_semantic,
                        override_shape=[config.eval.width, config.eval.height],
                    )["render"]
                    rendering = rendering / (rendering.norm(dim=0, keepdim=True) + 1e-8)
                    sim = torch.einsum("cq,qhw->chw", text_features, rendering)
                    label = sim.argmax(dim=0).cpu()

                # sem = render_palette(label, palette)
                # sem_gt = render_palette(label_img, palette)
                # torchvision.utils.save_image(sem, os.path.join(out_path, "{0:05d}".format(idx) + "_render.png"))
                # torchvision.utils.save_image(sem_gt, os.path.join(out_path, "{0:05d}".format(idx) + ".png"))
                confusion += metric.confusion_matrix(
                    label.cpu().numpy().reshape(-1), label_img.cpu().numpy().reshape(-1), config.scene.num_classes
                )

    metric.evaluate_confusion(confusion, stdout=True, dataset=config.scene.dataset_name)


def eval_seg_model(config, model_3d, label_mapping):
    eval_scene = os.listdir(config.model.model_dir)
    eval_scene.sort()

    model_2d_name = config.eval.model_2d.lower().replace("_", "")
    if model_2d_name == "openseg":
        from model.openseg_predictor import OpenSeg

        model_2d = OpenSeg("./weights/openseg_exported_clip", "ViT-L/14@336px")
    elif model_2d_name == "opensam":
        from model.opensam_predictor import OpenSAM

        model_2d = OpenSAM(
            "./weights/openseg_exported_clip", "./weights/groundingsam/sam_vit_h_4b8939.pth", "ViT-L/14@336px"
        )
    elif model_2d_name == "openfastsam":
        from model.openfastsam_predictor import OpenFastSAM

        model_2d = OpenFastSAM("./weights/openseg_exported_clip", "./weights/fastsam/FastSAM-x.pt", "ViT-L/14@336px")
    elif model_2d_name == "samclip":
        from model.samclip_predictor import SAMCLIP

        model_2d = SAMCLIP("./weights/groundingsam/sam_vit_h_4b8939.pth", "ViT-L/14@336px")
    elif model_2d_name == "fastsamclip":
        from model.fastsamclip_predictor import FastSAMCLIP

        model_2d = FastSAMCLIP("./weights/fastsam/FastSAM-x.pt", "ViT-L/14@336px")
    elif model_2d_name == "groundingsam":
        from model.groundingsam_predictor import GroundingSAM

        model_2d = GroundingSAM(
            "./weights/groundingsam/groundingdino_swint_ogc.pth",
            "./weights/groundingsam/sam_vit_h_4b8939.pth",
            "ViT-L/14@336px",
        )
    elif model_2d_name == "vlpart":
        from model.vlpart_predictor import VLPart

        model_2d = VLPart(
            "./weights/vlpart/swinbase_part_0a0000.pth",
            "./weights/vlpart/sam_vit_h_4b8939.pth",
            "ViT-L/14@336px",
        )

    confusion = np.zeros((config.scene.num_classes + 1, config.scene.num_classes), dtype=np.ulonglong)

    for i, scene_name in enumerate(tqdm(eval_scene)):
        if i > 20:
            break
        torch.cuda.empty_cache()
        with torch.no_grad():
            eval_config = deepcopy(config)
            eval_config.scene.scene_path = os.path.join(eval_config.scene.scene_path, scene_name)
            eval_config.model.model_dir = os.path.join(eval_config.model.model_dir, scene_name)
            scene = Scene(eval_config.scene)

            views = scene.getTrainCameras()
            # out_path = "eval_samples/"
            # os.makedirs(out_path, exist_ok=True)
            palette, text_features, mapping = get_text_features(model_2d, config.scene.dataset_name)
            for idx, view in enumerate(tqdm(views)):
                if idx % 5 != 0:
                    continue
                view.cuda()
                gt_path = str(view.image_path)
                label_img = get_mapped_label(config, gt_path, label_mapping)
                if label_img is None:
                    continue
                mapped = np.ones_like(label_img) * config.scene.num_classes
                for i in range(len(mapping)):
                    mapped[label_img == mapping[i]] = i
                label_img = torch.from_numpy(mapped).int().cpu()

                features = model_2d.extract_image_feature(
                    gt_path,
                    [label_img.shape[0], label_img.shape[1]],
                ).float()
                sim = torch.einsum("cq,qhw->chw", text_features.cpu(), features.cpu())
                label = sim.argmax(dim=0)

                # sem = render_palette(label, palette)
                # sem_gt = render_palette(label_img, palette)
                # torchvision.utils.save_image(sem, os.path.join(out_path, "{0:05d}".format(idx) + "_render.png"))
                # torchvision.utils.save_image(sem_gt, os.path.join(out_path, "{0:05d}".format(idx) + ".png"))
                confusion += metric.confusion_matrix(
                    label.cpu().numpy().reshape(-1), label_img.cpu().numpy().reshape(-1), config.scene.num_classes
                )

    metric.evaluate_confusion(confusion, stdout=True, dataset=config.scene.dataset_name)


if __name__ == "__main__":
    config = OmegaConf.load("./config/eval.yaml")
    print(OmegaConf.to_yaml(config))

    set_seed(config.pipeline.seed)
    evaluate(config)

    # for iter in [5, 10]:
    #     config_copy = deepcopy(config)
    #     config_copy.distill.iteration = iter
    #     evaluate(config_copy)
