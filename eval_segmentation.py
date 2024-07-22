import os
import torch
import torchvision
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from omegaconf import OmegaConf

from model import GaussianModel, render_chn
from scene import Scene
from utils.system_utils import set_seed, searchForMaxIteration

from model.mink_unet import mink_unet
from model.render_utils import get_mapped_label, get_text_features, render_palette
from dataset.fusion_utils import Voxelizer
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


def evaluate(config):
    if config.scene.dataset_name == "scannet20":
        config.scene.num_classes = 19
        label_mapping = read_label_mapping("./dataset/scannet/scannetv2-labels.modified.tsv", label_to="scannetid")
    elif config.scene.dataset_name == "cocomap":
        config.scene.num_classes = 20
        label_mapping = read_label_mapping("./dataset/scannet/scannetv2-labels.modified.tsv", label_to="cocomapid")

    if config.distill.feature_type == "all":
        model_3d = mink_unet(in_channels=56, out_channels=768, D=3, arch=config.distill.model_3d).cuda()
    elif config.distill.feature_type == "color":
        model_3d = mink_unet(in_channels=48, out_channels=768, D=3, arch=config.distill.model_3d).cuda()

    ckpt_path = init_dir(config)
    model_3d.load_state_dict(torch.load(ckpt_path))

    if config.eval.eval_mode == "2d":
        eval_fusion(config, model_3d, label_mapping)
    elif config.eval.eval_mode == "3d":
        eval_mink(config, model_3d, label_mapping)
    elif config.eval.eval_mode == "2d_and_3d":
        eval_mink_and_fusion(config, model_3d, label_mapping)
    elif config.eval.eval_mode == "pretrained":
        eval_seg_model(config, model_3d, label_mapping)
    elif config.eval.eval_mode == "labelmap":
        eval_labelmap(config, model_3d, label_mapping)


def eval_mink(config, model_3d, label_mapping):
    eval_scene = os.listdir(config.model.model_dir)
    eval_scene.sort()

    model_2d_name = config.distill.text_model.lower().replace("_", "")
    if model_2d_name == "lseg":
        from model.lseg_predictor import LSeg

        model_2d = LSeg(None)
    else:
        from model.openseg_predictor import OpenSeg

        model_2d = OpenSeg(None, "ViT-L/14@336px")

    bg_color = [1] * model_2d.embedding_dim if config.scene.white_background else [0] * model_2d.embedding_dim
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    confusion = np.zeros((config.scene.num_classes + 1, config.scene.num_classes), dtype=np.ulonglong)

    for i, scene_name in enumerate(tqdm(eval_scene)):
        torch.cuda.empty_cache()
        with torch.no_grad():
            eval_config = deepcopy(config)
            eval_config.scene.scene_path = os.path.join(eval_config.scene.scene_path, scene_name)
            eval_config.model.model_dir = os.path.join(eval_config.model.model_dir, scene_name)
            scene = Scene(eval_config.scene)
            gaussians = GaussianModel(eval_config.model.sh_degree)
            loaded_iter = config.model.load_iteration
            if loaded_iter == -1:
                loaded_iter = searchForMaxIteration(os.path.join(eval_config.model.model_dir, "point_cloud"))
            gaussians.load_ply(
                os.path.join(
                    eval_config.model.model_dir,
                    "point_cloud",
                    f"iteration_{loaded_iter}",
                    "point_cloud.ply",
                )
            )
            gaussians.create_semantic(model_2d.embedding_dim)

            locs, features = gaussians.get_locs_and_features(eval_config.distill.feature_type)
            voxelizer = Voxelizer(voxel_size=config.distill.voxel_size)
            locs, features, _, _, vox_ind = voxelizer.voxelize(locs, features, None, return_ind=True)
            locs = torch.from_numpy(locs).int()
            locs = torch.cat([torch.ones(locs.shape[0], 1, dtype=torch.int), locs], dim=1)
            features = torch.from_numpy(features).float()
            vox_ind = torch.from_numpy(vox_ind).cuda()

            sinput = ME.SparseTensor(features.cuda(), locs.cuda())
            output = model_3d(sinput).F[:, model_2d.embedding_dim * 0 : model_2d.embedding_dim * 1]
            output /= output.norm(dim=-1, keepdim=True) + 1e-8

            views = scene.getTrainCameras()
            out_path = os.path.join("eval_render", scene_name)
            os.makedirs(os.path.join(out_path, "gt"), exist_ok=True)
            os.makedirs(os.path.join(out_path, "3d"), exist_ok=True)  
            gaussians._features_semantic[vox_ind] = output
            palette, text_features = get_text_features(model_2d, dataset_name=config.scene.dataset_name)
            for idx, view in enumerate(views):
                # if idx % 5 != 0:
                #     continue
                view.cuda()
                gt_path = str(view.image_path)
                label_img = get_mapped_label(config, gt_path, label_mapping)
                if label_img is None:
                    continue
                label_img = torch.from_numpy(label_img).int().cpu()

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
                    label = rendering[1:].argmax(dim=0).cpu()
                else:
                    rendering = render_chn(
                        view,
                        gaussians,
                        eval_config.pipeline,
                        background,
                        num_channels=model_2d.embedding_dim,
                        override_color=gaussians._features_semantic,
                        override_shape=[config.eval.width, config.eval.height],
                    )["render"]
                    rendering = rendering / (rendering.norm(dim=0, keepdim=True) + 1e-8)
                    sim = torch.einsum("cq,qhw->chw", text_features, rendering)
                    label = sim[1:].argmax(dim=0).cpu()
                
                label += 1
                sem = render_palette(label, palette)
                sem_gt = render_palette(label_img, palette)
                torchvision.utils.save_image(sem, os.path.join(out_path, "3d", f"{views.camera_info[idx].image_name}.jpg"))
                torchvision.utils.save_image(sem_gt, os.path.join(out_path, "gt", f"{views.camera_info[idx].image_name}.jpg"))
                confusion += metric.confusion_matrix(
                    label.cpu().numpy().reshape(-1), label_img.cpu().numpy().reshape(-1), config.scene.num_classes
                )

    metric.evaluate_confusion(confusion, stdout=True, dataset=config.scene.dataset_name)


def eval_fusion(config, model_3d, label_mapping):
    eval_scene = os.listdir(config.model.model_dir)
    eval_scene.sort()

    model_2d_name = config.fusion.model_2d.lower().replace("_", "")
    if model_2d_name == "lseg":
        from model.lseg_predictor import LSeg

        model_2d = LSeg(None)
    else:
        from model.openseg_predictor import OpenSeg

        model_2d = OpenSeg(None, "ViT-L/14@336px")

    bg_color = [1] * model_2d.embedding_dim if config.scene.white_background else [0] * model_2d.embedding_dim
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    confusion = np.zeros((config.scene.num_classes + 1, config.scene.num_classes), dtype=np.ulonglong)

    for i, scene_name in enumerate(tqdm(eval_scene)):
        torch.cuda.empty_cache()
        with torch.no_grad():
            eval_config = deepcopy(config)
            eval_config.scene.scene_path = os.path.join(eval_config.scene.scene_path, scene_name)
            eval_config.model.model_dir = os.path.join(eval_config.model.model_dir, scene_name)
            scene = Scene(eval_config.scene)
            gaussians = GaussianModel(eval_config.model.sh_degree)
            loaded_iter = config.model.load_iteration
            if loaded_iter == -1:
                loaded_iter = searchForMaxIteration(os.path.join(eval_config.model.model_dir, "point_cloud"))
            gaussians.load_ply(
                os.path.join(
                    eval_config.model.model_dir,
                    "point_cloud",
                    f"iteration_{loaded_iter}",
                    "point_cloud.ply",
                )
            )
            gaussians.create_semantic(model_2d.embedding_dim)

            feature_path = os.path.join(eval_config.fusion.out_dir, scene_name, "0.pt")
            gt = torch.load(feature_path)
            feat, mask_full = gt["feat"], gt["mask_full"]

            views = scene.getTrainCameras()
            out_path = os.path.join("eval_render", scene_name)
            os.makedirs(os.path.join(out_path, "gt"), exist_ok=True)
            os.makedirs(os.path.join(out_path, "2d"), exist_ok=True)  
            gaussians._features_semantic[mask_full] = feat.float().cuda()
            palette, text_features = get_text_features(model_2d, dataset_name=config.scene.dataset_name)
            for idx, view in enumerate(views):
                # if idx % 5 != 0:
                #     continue
                view.cuda()
                gt_path = str(view.image_path)
                label_img = get_mapped_label(config, gt_path, label_mapping)
                if label_img is None:
                    continue
                label_img = torch.from_numpy(label_img).int().cpu()

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
                    label = rendering[1:].argmax(dim=0).cpu()
                else:
                    rendering = render_chn(
                        view,
                        gaussians,
                        eval_config.pipeline,
                        background,
                        num_channels=model_2d.embedding_dim,
                        override_color=gaussians._features_semantic,
                        override_shape=[config.eval.width, config.eval.height],
                    )["render"]
                    rendering = rendering / (rendering.norm(dim=0, keepdim=True) + 1e-8)
                    sim = torch.einsum("cq,qhw->chw", text_features, rendering)
                    label = sim[1:].argmax(dim=0).cpu()

                label += 1
                sem = render_palette(label, palette)
                sem_gt = render_palette(label_img, palette)
                torchvision.utils.save_image(sem, os.path.join(out_path, "2d", f"{views.camera_info[idx].image_name}.jpg"))
                torchvision.utils.save_image(sem_gt, os.path.join(out_path, "gt", f"{views.camera_info[idx].image_name}.jpg"))
                confusion += metric.confusion_matrix(
                    label.cpu().numpy().reshape(-1), label_img.cpu().numpy().reshape(-1), config.scene.num_classes
                )

    metric.evaluate_confusion(confusion, stdout=True, dataset=config.scene.dataset_name)


def eval_mink_and_fusion(config, model_3d, label_mapping):
    eval_scene = os.listdir(config.model.model_dir)
    eval_scene.sort()

    performance_dict = {}

    model_2d_name = config.fusion.model_2d.lower().replace("_", "")
    if model_2d_name == "lseg":
        from model.lseg_predictor import LSeg

        model_2d = LSeg(None)
    else:
        from model.openseg_predictor import OpenSeg

        model_2d = OpenSeg(None, "ViT-L/14@336px")

    text_model_name = config.distill.text_model.lower().replace("_", "")
    if text_model_name == model_2d_name:
        text_model = model_2d
    elif text_model_name == "lseg":
        from model.lseg_predictor import LSeg

        text_model = LSeg(None)
    else:
        from model.openseg_predictor import OpenSeg

        text_model = OpenSeg(None, "ViT-L/14@336px")

    bg_color = [1] * model_2d.embedding_dim if config.scene.white_background else [0] * model_2d.embedding_dim
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    confusion = np.zeros((config.scene.num_classes + 1, config.scene.num_classes), dtype=np.ulonglong)

    for i, scene_name in enumerate(tqdm(eval_scene)):
        torch.cuda.empty_cache()
        with torch.no_grad():
            eval_config = deepcopy(config)
            eval_config.scene.scene_path = os.path.join(eval_config.scene.scene_path, scene_name)
            eval_config.model.model_dir = os.path.join(eval_config.model.model_dir, scene_name)
            scene = Scene(eval_config.scene)
            gaussians = GaussianModel(eval_config.model.sh_degree)
            loaded_iter = config.model.load_iteration
            if loaded_iter == -1:
                loaded_iter = searchForMaxIteration(os.path.join(eval_config.model.model_dir, "point_cloud"))
            gaussians.load_ply(
                os.path.join(
                    eval_config.model.model_dir,
                    "point_cloud",
                    f"iteration_{loaded_iter}",
                    "point_cloud.ply",
                )
            )
            gaussians.create_semantic(model_2d.embedding_dim + text_model.embedding_dim)

            locs, features = gaussians.get_locs_and_features(eval_config.distill.feature_type)
            voxelizer = Voxelizer(voxel_size=config.distill.voxel_size)
            locs, features, _, _, vox_ind = voxelizer.voxelize(locs, features, None, return_ind=True)
            locs = torch.from_numpy(locs).int()
            locs = torch.cat([torch.ones(locs.shape[0], 1, dtype=torch.int), locs], dim=1)
            features = torch.from_numpy(features).float()
            vox_ind = torch.from_numpy(vox_ind).cuda()

            sinput = ME.SparseTensor(features.cuda(), locs.cuda())
            output = model_3d(sinput).F[:, text_model.embedding_dim * 0 : text_model.embedding_dim * 1]

            feature_path = os.path.join(eval_config.fusion.out_dir, scene_name, "0.pt")
            gt = torch.load(feature_path)
            feat, mask_full = gt["feat"].float(), gt["mask_full"]

            views = scene.getTrainCameras()
            out_path = os.path.join("eval_render", scene_name)
            os.makedirs(os.path.join(out_path, "gt"), exist_ok=True)
            os.makedirs(os.path.join(out_path, "2d_and_3d"), exist_ok=True)
            feat /= feat.norm(dim=-1, keepdim=True) + 1e-8
            output /= output.norm(dim=-1, keepdim=True) + 1e-8
            gaussians._features_semantic[mask_full, :model_2d.embedding_dim] = feat.float().cuda()
            gaussians._features_semantic[vox_ind, model_2d.embedding_dim:] = output
            palette, text_features_2d = get_text_features(model_2d, dataset_name=config.scene.dataset_name)
            palette, text_features_3d = get_text_features(text_model, dataset_name=config.scene.dataset_name)

            for idx, view in enumerate(views):
                # if idx % 5 != 0:
                #     continue
                view.cuda()
                gt_path = str(view.image_path)
                label_img = get_mapped_label(config, gt_path, label_mapping)
                if label_img is None:
                    continue
                label_img = torch.from_numpy(label_img).int().cpu()

                if config.eval.feature_fusion == "concat":
                    cat_text_features = torch.cat([text_features_2d, text_features_3d], dim=1)
                    if config.eval.pred_on_3d:
                        sim = torch.einsum("cq,dq->dc", cat_text_features, gaussians._features_semantic)
                        label_soft = sim.softmax(dim=1)
                        rendering = render_chn(
                            view,
                            gaussians,
                            eval_config.pipeline,
                            background,
                            num_channels=label_soft.shape[1],
                            override_color=label_soft,
                            override_shape=[config.eval.width, config.eval.height],
                        )["render"]
                        label = rendering[1:].argmax(dim=0).cpu()
                    else:
                        rendering1 = render_chn(
                            view,
                            gaussians,
                            eval_config.pipeline,
                            background,
                            num_channels=model_2d.embedding_dim,
                            override_color=gaussians._features_semantic[:, :model_2d.embedding_dim],
                            override_shape=[config.eval.width, config.eval.height],
                        )["render"]
                        rendering2 = render_chn(
                            view,
                            gaussians,
                            eval_config.pipeline,
                            background,
                            num_channels=text_model.embedding_dim,
                            override_color=gaussians._features_semantic[:, model_2d.embedding_dim:],
                            override_shape=[config.eval.width, config.eval.height],
                        )["render"]
                        rendering = torch.cat([rendering1, rendering2], dim=0)
                        rendering = rendering / (rendering.norm(dim=0, keepdim=True) + 1e-8)
                        sim = torch.einsum("cq,qhw->chw", cat_text_features, rendering)
                        label = sim[1:].argmax(dim=0).cpu()
                elif config.eval.feature_fusion == "argmax":
                    if config.eval.pred_on_3d:
                        sim_2d = torch.einsum(
                            "cq,dq->dc", text_features_2d, gaussians._features_semantic[:, :model_2d.embedding_dim]
                        )
                        sim_3d = torch.einsum(
                            "cq,dq->dc", text_features_3d, gaussians._features_semantic[:, model_2d.embedding_dim:]
                        )
                        sim = torch.stack([sim_2d, sim_3d], dim=1)
                        label_soft = sim.max(dim=1).values.softmax(dim=1)
                        rendering = render_chn(
                            view,
                            gaussians,
                            eval_config.pipeline,
                            background,
                            num_channels=label_soft.shape[1],
                            override_color=label_soft,
                            override_shape=[config.eval.width, config.eval.height],
                        )["render"]
                        label = rendering[1:].argmax(dim=0).cpu()
                    else:
                        rendering1 = render_chn(
                            view,
                            gaussians,
                            eval_config.pipeline,
                            background,
                            num_channels=model_2d.embedding_dim,
                            override_color=gaussians._features_semantic[:, :model_2d.embedding_dim],
                            override_shape=[config.eval.width, config.eval.height],
                        )["render"]
                        rendering2 = render_chn(
                            view,
                            gaussians,
                            eval_config.pipeline,
                            background,
                            num_channels=text_model.embedding_dim,
                            override_color=gaussians._features_semantic[:, model_2d.embedding_dim:],
                            override_shape=[config.eval.width, config.eval.height],
                        )["render"]
                        rendering1 = rendering1 / (rendering1.norm(dim=0, keepdim=True) + 1e-8)
                        rendering2 = rendering2 / (rendering2.norm(dim=0, keepdim=True) + 1e-8)
                        sim_2d = torch.einsum("cq,qhw->chw", text_features_2d, rendering1)
                        sim_3d = torch.einsum("cq,qhw->chw", text_features_3d, rendering2)
                        sim = torch.stack([sim_2d, sim_3d], dim=1)
                        label = sim[1:].max(dim=1).values.argmax(dim=0).cpu()

                label += 1
                sem = render_palette(label, palette)
                sem_gt = render_palette(label_img, palette)
                torchvision.utils.save_image(
                    sem, os.path.join(out_path, "2d_and_3d", f"{views.camera_info[idx].image_name}.jpg")
                )
                torchvision.utils.save_image(
                    sem_gt, os.path.join(out_path, "gt", f"{views.camera_info[idx].image_name}.jpg")
                )
                confusion_img = metric.confusion_matrix(
                    label.cpu().numpy().reshape(-1), label_img.cpu().numpy().reshape(-1), config.scene.num_classes
                )
                confusion += confusion_img

    metric.evaluate_confusion(confusion, stdout=True, dataset=config.scene.dataset_name)


def eval_seg_model(config, model_3d, label_mapping):
    eval_scene = os.listdir(config.model.model_dir)
    eval_scene.sort()

    model_2d_name = config.fusion.model_2d.lower().replace("_", "")
    if model_2d_name == "openseg":
        from model.openseg_predictor import OpenSeg

        model_2d = OpenSeg("./weights/openseg_exported_clip", "ViT-L/14@336px")
    elif model_2d_name == "lseg":
        from model.lseg_predictor import LSeg

        model_2d = LSeg("./weights/lseg/demo_e200.ckpt")
    elif model_2d_name == "samclip":
        from model.samclip_predictor import SAMCLIP

        model_2d = SAMCLIP("./weights/groundingsam/sam_vit_h_4b8939.pth", "ViT-L/14@336px")
    elif model_2d_name == "vlpart":
        from model.vlpart_predictor import VLPart

        model_2d = VLPart(
            "./weights/vlpart/swinbase_part_0a0000.pth",
            "./weights/vlpart/sam_vit_h_4b8939.pth",
            "ViT-L/14@336px",
        )

    confusion = np.zeros((config.scene.num_classes + 1, config.scene.num_classes), dtype=np.ulonglong)

    for i, scene_name in enumerate(tqdm(eval_scene)):
        torch.cuda.empty_cache()
        with torch.no_grad():
            eval_config = deepcopy(config)
            eval_config.scene.scene_path = os.path.join(eval_config.scene.scene_path, scene_name)
            eval_config.model.model_dir = os.path.join(eval_config.model.model_dir, scene_name)
            scene = Scene(eval_config.scene)

            views = scene.getTrainCameras()
            out_path = os.path.join("eval_render", scene_name)
            os.makedirs(os.path.join(out_path, "gt"), exist_ok=True)
            os.makedirs(os.path.join(out_path, model_2d_name), exist_ok=True)
            palette, text_features = get_text_features(model_2d, config.scene.dataset_name)
            for idx, view in enumerate(tqdm(views)):
                # if idx % 5 != 0:
                #     continue
                view.cuda()
                gt_path = str(view.image_path)
                label_img = get_mapped_label(config, gt_path, label_mapping)
                if label_img is None:
                    continue
                label_img = torch.from_numpy(label_img).int().cpu()

                features = model_2d.extract_image_feature(
                    gt_path,
                    [label_img.shape[0], label_img.shape[1]],
                ).float()
                sim = torch.einsum("cq,qhw->chw", text_features.cpu(), features.cpu())
                label = sim.argmax(dim=0)

                sem = render_palette(label, palette)
                sem_gt = render_palette(label_img, palette)
                torchvision.utils.save_image(
                    sem, os.path.join(out_path, model_2d_name, f"{views.camera_info[idx].image_name}.jpg")
                )
                torchvision.utils.save_image(
                    sem_gt, os.path.join(out_path, "gt", f"{views.camera_info[idx].image_name}.jpg")
                )
                confusion += metric.confusion_matrix(
                    label.cpu().numpy().reshape(-1), label_img.cpu().numpy().reshape(-1), config.scene.num_classes
                )

    metric.evaluate_confusion(confusion, stdout=True, dataset=config.scene.dataset_name)


def eval_labelmap(config, model_3d, label_mapping):
    eval_scene = os.listdir(config.model.model_dir)
    eval_scene.sort()

    confusion = np.zeros((config.scene.num_classes + 1, config.scene.num_classes), dtype=np.ulonglong)
    from model.openseg_predictor import OpenSeg

    model_2d = OpenSeg(None, "ViT-L/14@336px")

    for i, scene_name in enumerate(tqdm(eval_scene)):
        torch.cuda.empty_cache()
        with torch.no_grad():
            eval_config = deepcopy(config)
            eval_config.scene.scene_path = os.path.join(eval_config.scene.scene_path, scene_name)
            eval_config.model.model_dir = os.path.join(eval_config.model.model_dir, scene_name, "2")
            scene = Scene(eval_config.scene)

            views = scene.getTrainCameras()
            out_path = os.path.join("eval_render_langsplat_2", scene_name)
            os.makedirs(os.path.join(out_path, "gt"), exist_ok=True)
            os.makedirs(os.path.join(out_path, "labelmap"), exist_ok=True)
            palette, _ = get_text_features(model_2d, config.scene.dataset_name)

            label_map_pts = os.listdir(eval_config.model.model_dir)
            for idx, view in enumerate(tqdm(views)):
                name = view.image_name.split(".")[0]
                if f"{name}.pt" not in label_map_pts:
                    continue

                gt_path = str(view.image_path)
                label_img = get_mapped_label(config, gt_path, label_mapping)
                if label_img is None:
                    continue
                label_img = torch.from_numpy(label_img).int().cpu()

                label = torch.load(os.path.join(eval_config.model.model_dir, f"{name}.pt"))
                label += 1

                sem = render_palette(label, palette)
                sem_gt = render_palette(label_img, palette)
                torchvision.utils.save_image(
                    sem, os.path.join(out_path, "labelmap", f"{views.camera_info[idx].image_name}.jpg")
                )
                torchvision.utils.save_image(
                    sem_gt, os.path.join(out_path, "gt", f"{views.camera_info[idx].image_name}.jpg")
                )
                confusion += metric.confusion_matrix(
                    label.cpu().numpy().reshape(-1), label_img.cpu().numpy().reshape(-1), config.scene.num_classes
                )

    metric.evaluate_confusion(confusion, stdout=True, dataset=config.scene.dataset_name)

if __name__ == "__main__":
    config = OmegaConf.load("./config/eval.yaml")
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)
    print(OmegaConf.to_yaml(config))

    set_seed(config.pipeline.seed)
    evaluate(config)