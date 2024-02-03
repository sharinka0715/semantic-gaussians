import os
import torch
import warnings
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from scene import Scene

warnings.filterwarnings("ignore")


def fuse_one_scene(config, model_2d):
    scene = Scene(config.scene)

    loader = DataLoader(
        scene.getTrainCameras(),
        batch_size=1,
        shuffle=False,
        collate_fn=lambda x: x,
        num_workers=config.fusion.num_workers,
    )

    # feature fusion
    with torch.no_grad():
        for idx, view in enumerate(tqdm(loader)):
            view = view[0]
            # Call seg model to get per-pixel features
            gt_path = view.image_path
            features = model_2d.extract_image_feature(
                gt_path,
                [config.fusion.img_dim[1], config.fusion.img_dim[0]],
            )

            os.makedirs(os.path.join(config.scene.scene_path, "sam_masks"), exist_ok=True)
            np.save(os.path.join(config.scene.scene_path, "sam_masks", f"{view.image_name}.npy"), features)


if __name__ == "__main__":
    config = OmegaConf.load("./config/fusion_scannet.yaml")
    override_config = OmegaConf.from_cli()
    config = OmegaConf.merge(config, override_config)
    print(OmegaConf.to_yaml(config))

    scenes = os.listdir(config.model.model_dir)
    scenes.sort()

    from model.sam_predictor import SAMAutoMask

    model_2d = SAMAutoMask("./weights/groundingsam/sam_vit_h_4b8939.pth")

    fuse_one_scene(config, model_2d)
    exit()

    for idx, scene in enumerate(tqdm(scenes)):
        if config.model.dynamic:
            T = len(os.listdir(os.path.join(config.scene.scene_path, scene)))
            for t in tqdm(range(T)):
                model_dir = os.path.join(config.model.model_dir, scene)
                scene_path = os.path.join(config.scene.scene_path, scene, str(t))
                out_dir = os.path.join(config.fusion.out_dir, scene)
                print(scene_path)

                scene_config = deepcopy(config)
                scene_config.scene.scene_path = scene_path
                scene_config.model.model_dir = model_dir
                scene_config.model.dynamic_t = t
                scene_config.fusion.out_dir = out_dir
                fuse_one_scene(scene_config, model_2d)
        else:
            model_dir = os.path.join(config.model.model_dir, scene)
            scene_path = os.path.join(config.scene.scene_path, scene)
            out_dir = os.path.join(config.fusion.out_dir, scene)

            scene_config = deepcopy(config)
            scene_config.scene.scene_path = scene_path
            scene_config.model.model_dir = model_dir
            scene_config.fusion.out_dir = out_dir
            fuse_one_scene(scene_config, model_2d)
