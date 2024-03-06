import os
import torch
import imageio
import numpy as np
import skimage.transform as sktf
from dataset.scannet.scannet_constants import (
    SCANNET_CLASS_LABELS_20,
    SCANNET_VALID_CLASS_IDS_20,
    SCANNET_COLOR_MAP_20,
    SCANNET_CLASS_LABELS_200,
    SCANNET_COLOR_MAP_200,
    SCANNET_VALID_CLASS_IDS_200,
)


def get_text_features(model_2d, dataset_name="scannet_20"):
    if isinstance(dataset_name, list):
        labelset = dataset_name
        valid_classes = SCANNET_VALID_CLASS_IDS_20
        color_map = SCANNET_COLOR_MAP_20
    elif dataset_name == "scannet_20":
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
    labelset = ["unlabeled"] + labelset
    scannet_palette = [torch.tensor(color_map[0])] + scannet_palette
    mapping = [0] + mapping

    # scannet_palette[9] = scannet_palette[8]

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
