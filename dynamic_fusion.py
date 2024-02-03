import os
import torch
import imageio
import warnings
import torchvision
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from omegaconf import OmegaConf
import random
import tensorflow as tf2
import tensorflow.compat.v1 as tf

from model import GaussianModel, render, render_sem
from scene import Scene
from utils.system_utils import searchForMaxIteration, set_seed
from model.render_utils import get_text_features, render_palette
from dataset.label_constant import SCANNET_LABELS_20, SCANNET_COLOR_MAP_20
from dataset.fusion_utils import PointCloudToImageMapper

warnings.filterwarnings("ignore")


def fuse_one_scene(config, model_2d):
    scene_config = deepcopy(config)
    scene_config.scene.scene_path = os.path.join(config.scene.scene_path, "0")
    scene = Scene(scene_config.scene)
    gaussians = GaussianModel(config.model.sh_degree)

    gaussians.load_dynamic_npz(os.path.join(config.model.model_dir, "params.npz"), 0)
    gaussians.create_semantic(768)

    bg_color = [1] * 768 if config.scene.white_background else [0] * 768
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    views = scene.getTrainCameras()

    # feature fusion
    T = len(os.listdir(config.scene.scene_path))

    with torch.no_grad():
        for t in tqdm(range(T)):
            if t % 15 != 0:
                continue
            torch.cuda.empty_cache()
            scene_config = deepcopy(config)
            scene_config.scene.scene_path = os.path.join(config.scene.scene_path, str(t))
            scene = Scene(scene_config.scene)
            views = scene.getTrainCameras()
            gaussians.load_dynamic_npz(os.path.join(config.model.model_dir, "params.npz"), t)
            vis_id = torch.zeros((gaussians._xyz.shape[0], len(views)), dtype=int)
            for idx, view in enumerate(tqdm(views)):
                view.cuda()
                mapper = PointCloudToImageMapper(
                    config.fusion.img_dim,
                    config.fusion.visibility_threshold,
                    config.fusion.cut_boundary,
                    views.camera_info[idx].intrinsics,
                )

                # Call seg model to get per-pixel features
                gt_path = view.image_path
                features = model_2d.extract_image_feature(
                    gt_path,
                    [config.fusion.img_dim[1], config.fusion.img_dim[0]],
                )

                # palette, text_features, _ = get_text_features(
                #     model_2d,
                #     # config.scene.dataset_name,
                #     [
                #         "basketball",
                #         "a person head",
                #         "a person arm",
                #         "a person hand",
                #         "a person leg",
                #         "a person foot",
                #         "a person torso",
                #     ],
                # )
                # sim = torch.einsum("cq,qhw->chw", text_features, features.float().cuda())
                # label = sim.argmax(dim=0)

                # new_3d = torch.zeros((label.shape[0], label.shape[1], 3)).cuda()
                # u_index = torch.unique(label)
                # for index in u_index:
                #     new_3d[label == index] = torch.tensor(
                #         [
                #             palette[index * 3] / 255.0,
                #             palette[index * 3 + 1] / 255.0,
                #             palette[index * 3 + 2] / 255.0,
                #         ]
                #     ).cuda()
                # torchvision.utils.save_image(
                #     new_3d.permute(2, 0, 1),
                #     "semantic/{0:05d}.png".format(idx),
                # )

                if config.fusion.depth == "image":
                    depth_path = os.path.join(config.scene.scene_path, "depth", view.image_name + ".png")
                    depth = imageio.v2.imread(depth_path) / config.fusion.depth_scale
                elif config.fusion.depth == "render":
                    depth = (
                        render(
                            view,
                            gaussians,
                            config.pipeline,
                            background,
                            override_shape=config.fusion.img_dim,
                        )["depth"]
                        .cpu()
                        .numpy()[0]
                    )
                elif config.fusion.depth == "surface":
                    depth = "surface"
                else:
                    depth = None

                # calculate the 3d-2d mapping based on the depth
                mapping = np.ones([gaussians._xyz.shape[0], 4], dtype=int)
                mapping[:, 1:4], weight = mapper.compute_mapping(
                    view.world_view_transform.cpu().numpy(),
                    gaussians._xyz.cpu().numpy(),
                    depth,
                )
                if mapping[:, 3].sum() == 0:  # no points corresponds to this image, skip
                    continue

                mapping = torch.from_numpy(mapping)
                mask = mapping[:, 3]
                vis_id[:, idx] = mask
                features_mapping = features[:, mapping[:, 1], mapping[:, 2]]
                features_mapping = features_mapping.permute(1, 0).cuda()

                # cs = torch.nn.functional.cosine_similarity(gaussians._features_semantic, features_mapping, dim=-1).cpu()

                mask_zero = (mask != 0) & (gaussians._times[:, 0].cpu() == 0)
                gaussians._features_semantic[mask_zero] += features_mapping[mask_zero]
                gaussians._times[mask_zero] += 1

                mask_one = (mask != 0) & (
                    gaussians._times[:, 0].cpu() != 0
                )  # & (cs >= config.fusion.outlier_threshold)
                gaussians._times[mask_one] += 1
                gaussians._features_semantic[mask_one] += (
                    features_mapping[mask_one] - gaussians._features_semantic[mask_one]
                ) / gaussians._times[mask_one]

            gaussians._times[gaussians._times == 0] = 1e-5
            gaussians._features_semantic /= gaussians._times
            point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])

            palette, text_features, _ = get_text_features(model_2d, dataset_name=config.scene.dataset_name)
            palette, text_features, _ = get_text_features(
                model_2d,
                [
                    "basketball",
                    "a person head",
                    "a person arm",
                    "a person hand",
                    "a person leg",
                    "a person foot",
                    "a person torso",
                ],
            )
            sim = torch.einsum("cq,dq->dc", text_features, gaussians._features_semantic)
            label = sim.argmax(dim=1)

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

            from utils.sh_utils import RGB2SH

            bg_color = [1, 1, 1] if config.scene.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            gaussians._features_dc[:, 0] = RGB2SH(new_3d)
            gaussians._features_rest[:] = 0
            for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
                view.cuda()
                rendering = render(
                    view,
                    gaussians,
                    config.pipeline,
                    background,
                    override_shape=config.fusion.img_dim,
                )["render"]
                torchvision.utils.save_image(rendering, os.path.join("semantic", "{0:05d}".format(idx) + ".png"))
            break

        # palette, text_features, _ = get_text_features(
        #     model_2d,
        #     [
        #         "a guitar headstock",
        #         "a guitar fingerboard",
        #         "a guitar hole",
        #         "a guitar body",
        #         "a guitar bridge",
        #     ],
        # )  # dataset_name=config.scene.dataset_name)
        # for idx, view in enumerate(tqdm(views)):
        #     rendering = render_sem(
        #         view,
        #         gaussians,
        #         config.pipeline,
        #         background,
        #         override_color=gaussians._features_semantic,
        #         override_shape=config.fusion.img_dim,
        #     )["render"]
        #     rendering = rendering / (rendering.norm(dim=0, keepdim=True) + 1e-8)
        #     sim = torch.einsum("cq,qhw->chw", text_features, rendering)
        #     label = sim.argmax(dim=0).cpu()
        #     sem = render_palette(label, palette)
        #     torchvision.utils.save_image(sem, os.path.join("semantic", "{0:05d}".format(idx) + ".png"))
        # exit()

    # save fused features
    os.makedirs(config.fusion.out_dir, exist_ok=True)
    for n in range(config.fusion.num_rand_file_per_scene):
        if gaussians._xyz.shape[0] < config.fusion.n_split_points:
            n_points_cur = gaussians._xyz.shape[0]  # to handle point cloud numbers less than n_split_points
        else:
            n_points_cur = config.fusion.n_split_points

        rand_ind = np.random.choice(range(gaussians._xyz.shape[0]), n_points_cur, replace=False)

        mask_entire = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool)
        mask_entire[rand_ind] = True
        mask = torch.zeros(gaussians._xyz.shape[0], dtype=torch.bool)
        mask[point_ids] = True
        mask_entire = mask_entire & mask

        torch.save(
            {
                "feat": gaussians._features_semantic[mask_entire].cpu().half(),
                "mask_full": mask_entire,
            },
            os.path.join(config.fusion.out_dir + "/%d.pt" % (n)),
        )


if __name__ == "__main__":
    config = OmegaConf.load("./config/fusion_panoptic.yaml")
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)
    print(OmegaConf.to_yaml(config))

    set_seed(config.pipeline.seed)

    scenes = os.listdir(config.model.model_dir)
    scenes.sort()

    model_2d_name = config.fusion.model_2d.lower().replace("_", "")
    if model_2d_name == "openseg":
        from model.openseg_predictor import OpenSeg

        model_2d = OpenSeg("./weights/openseg_exported_clip", "ViT-L/14@336px")
    elif model_2d_name == "opensam":
        from model.opensam_predictor import OpenSAM

        model_2d = OpenSAM(
            "./weights/openseg_exported_clip", "./weights/groundingsam/sam_vit_h_4b8939.pth", "ViT-L/14@336px"
        )
    elif model_2d_name == "samclip":
        from model.samclip_predictor import SAMCLIP

        model_2d = SAMCLIP("./weights/groundingsam/sam_vit_h_4b8939.pth", "ViT-L/14@336px")
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

    fuse_one_scene(config, model_2d)
    exit()

    for idx, scene in enumerate(tqdm(scenes)):
        model_dir = os.path.join(config.model.model_dir, scene)
        scene_path = os.path.join(config.scene.scene_path, scene)
        out_dir = os.path.join(config.fusion.out_dir, scene)

        scene_config = deepcopy(config)
        scene_config.scene.scene_path = scene_path
        scene_config.model.model_dir = model_dir
        scene_config.fusion.out_dir = out_dir
        fuse_one_scene(scene_config, model_2d)
