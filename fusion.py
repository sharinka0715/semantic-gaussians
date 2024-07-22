import os
import torch
import imageio
import warnings
import torchvision
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from omegaconf import OmegaConf
import skimage.transform as sktf
from torch.utils.data import DataLoader

from model import GaussianModel, render, render_chn
from scene import Scene
from utils.system_utils import searchForMaxIteration, set_seed
from model.render_utils import get_text_features, render_palette
from dataset.fusion_utils import PointCloudToImageMapper
from dataset.scannet.scannet_constants import SCANNET20_CLASS_LABELS

warnings.filterwarnings("ignore")


def fuse_one_scene(config, model_2d):
    scene = Scene(config.scene)
    gaussians = GaussianModel(config.model.sh_degree)

    if config.model.dynamic:
        gaussians.load_dynamic_npz(os.path.join(config.model.model_dir, "params.npz"), config.model.dynamic_t)
    else:
        loaded_iter = config.model.load_iteration
        if loaded_iter == -1:
            loaded_iter = searchForMaxIteration(os.path.join(config.model.model_dir, "point_cloud"))
        print(f"Loading iteration {loaded_iter}...")
        gaussians.load_ply(
            os.path.join(
                config.model.model_dir,
                "point_cloud",
                f"iteration_{loaded_iter}",
                "point_cloud.ply",
            )
        )

    gaussians.create_semantic(model_2d.embedding_dim)

    bg_color = [1] * model_2d.embedding_dim if config.scene.white_background else [0] * model_2d.embedding_dim
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    views = scene.getTrainCameras()

    loader = DataLoader(
        views,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=config.fusion.num_workers,
    )

    # feature fusion
    with torch.no_grad():
        vis_id = torch.zeros((gaussians._xyz.shape[0], len(views)), dtype=int)
        for idx, view in enumerate(tqdm(loader)):
            if idx % 5 != 0:
                continue
            view = view[0]
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

            ################################# Save relevancy maps from pretrained models ####################################
            # pretrained_save_path = os.path.join(config.model.model_dir, "render", "pretrained")
            # os.makedirs(pretrained_save_path, exist_ok=True)

            # palette, text_features = get_text_features(
            #     model_2d,
            #     model_2d.classes.split("."),
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
            #     "{}/{}.jpg".format(pretrained_save_path, views.camera_info[idx].image_name),
            # )
            #################################################################################################################

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

            mask_k = mask != 0
            gaussians._times[mask_k] += 1
            gaussians._features_semantic[mask_k] += features_mapping[mask_k]

        gaussians._times[gaussians._times == 0] = 1e-5
        gaussians._features_semantic /= gaussians._times
        point_ids = torch.unique(vis_id.nonzero(as_tuple=False)[:, 0])

        ################################ Save relevancy maps from projected gaussians #################################
        # relevancy_save_path = os.path.join(config.model.model_dir, "render", "relevancy")
        # os.makedirs(relevancy_save_path, exist_ok=True)
        # palette, text_features = get_text_features(
        #     model_2d,
        #     model_2d.classes.split("."),
        # )
        # sim = torch.einsum("cq,dq->dc", text_features, gaussians._features_semantic)
        # label = sim.argmax(dim=1)

        # new_3d = torch.zeros((label.shape[0], 3)).cuda()
        # u_index = torch.unique(label)
        # for index in u_index:
        #     new_3d[label == index] = torch.tensor(
        #         [
        #             palette[index * 3] / 255.0,
        #             palette[index * 3 + 1] / 255.0,
        #             palette[index * 3 + 2] / 255.0,
        #         ]
        #     ).cuda()

        # for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        #     view.cuda()
        #     rendering = render(
        #         view,
        #         gaussians,
        #         config.pipeline,
        #         background,
        #         override_shape=config.fusion.img_dim,
        #         override_color=new_3d,
        #     )["render"]
        #     torchvision.utils.save_image(
        #         rendering,
        #         os.path.join(relevancy_save_path, views.camera_info[idx].image_name + ".jpg"),
        #     )
        #################################################################################################################

        ################################ Save semantic maps from projected gaussians #################################
        # semantic_save_path = os.path.join(config.model.model_dir, "render", "semantic")
        # os.makedirs(semantic_save_path, exist_ok=True)
        # palette, text_features = get_text_features(
        #     model_2d,
        #     model_2d.classes.split("."),
        # )
        # sim = torch.einsum("cq,dq->dc", text_features.float(), gaussians._features_semantic)
        # label_soft = sim.softmax(dim=1)
        # label_hard = torch.nn.functional.one_hot(sim.argmax(dim=1), num_classes=label_soft.shape[1]).float()

        # for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        #     view.cuda()
        #     label = (
        #         render_chn(
        #             view,
        #             gaussians,
        #             config.pipeline,
        #             background,
        #             override_shape=config.fusion.img_dim,
        #             num_channels=label_soft.shape[1],
        #             override_color=label_hard,
        #         )["render"]
        #         .argmax(dim=0)
        #         .cpu()
        #     )

        #     rendering = render_palette(label, palette)
        #     torchvision.utils.save_image(
        #         rendering,
        #         os.path.join(semantic_save_path, views.camera_info[idx].image_name + ".jpg"),
        #     )
        #################################################################################################################
        # exit()

    # save fused features
    if config.model.dynamic:
        os.makedirs(config.fusion.out_dir + "/%d" % config.model.dynamic_t, exist_ok=True)
    else:
        os.makedirs(config.fusion.out_dir, exist_ok=True)
    for n in range(config.fusion.num_rand_file_per_scene):
        save_path = (
            os.path.join(config.fusion.out_dir + "/%d/%d.pt" % (config.model.dynamic_t, n))
            if config.model.dynamic
            else os.path.join(config.fusion.out_dir + "/%d.pt" % (n))
        )
        if gaussians._xyz.shape[0] < config.fusion.n_split_points: # to handle point cloud numbers less than n_split_points
            torch.save(
            {
                "feat": gaussians._features_semantic.cpu().half(),
                "mask_full": torch.ones(gaussians._xyz.shape[0], dtype=torch.bool),
            },
            save_path
        )
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
                save_path
            )


if __name__ == "__main__":
    config = OmegaConf.load("./config/fusion_scannet.yaml")
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)
    print(OmegaConf.to_yaml(config))

    set_seed(config.pipeline.seed)

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

    scenes = os.listdir(config.model.model_dir)
    scenes.sort()
    model_2d.set_predefined_cls(SCANNET20_CLASS_LABELS)

    fuse_one_scene(config, model_2d)

    # for idx, scene in enumerate(tqdm(scenes)):
    #     if config.model.dynamic:
    #         T = len(os.listdir(os.path.join(config.scene.scene_path, scene)))
    #         for t in tqdm(range(T)):
    #             model_dir = os.path.join(config.model.model_dir, scene)
    #             scene_path = os.path.join(config.scene.scene_path, scene, str(t))
    #             out_dir = os.path.join(config.fusion.out_dir, scene)
    #             print(scene_path)

    #             scene_config = deepcopy(config)
    #             scene_config.scene.scene_path = scene_path
    #             scene_config.model.model_dir = model_dir
    #             scene_config.model.dynamic_t = t
    #             scene_config.fusion.out_dir = out_dir
    #             fuse_one_scene(scene_config, model_2d)
    #     else:
    #         model_dir = os.path.join(config.model.model_dir, scene)
    #         scene_path = os.path.join(config.scene.scene_path, scene)
    #         out_dir = os.path.join(config.fusion.out_dir, scene)

    #         scene_config = deepcopy(config)
    #         scene_config.scene.scene_path = scene_path
    #         scene_config.model.model_dir = model_dir
    #         scene_config.fusion.out_dir = out_dir
    #         fuse_one_scene(scene_config, model_2d)
