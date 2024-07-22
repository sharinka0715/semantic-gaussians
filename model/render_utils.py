import os
import torch
import imageio
import numpy as np
import skimage.transform as sktf
from dataset.scannet.scannet_constants import SCANNET20_CLASS_LABELS, COCOMAP_CLASS_LABELS, COLORMAP


def get_text_features(model_2d, dataset_name="scannet20"):
    if isinstance(dataset_name, list):
        labelset = dataset_name
    elif dataset_name == "scannet20":
        labelset = list(SCANNET20_CLASS_LABELS)
    elif dataset_name == "cocomap":
        labelset = list(COCOMAP_CLASS_LABELS)

    # add unlabeled label and palette
    labelset = ["other"] + labelset

    palette = torch.tensor(COLORMAP[:len(labelset)+1]).cuda().flatten()
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


def get_mapped_label(config, image_path, label_mapping):
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
