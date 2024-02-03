import os
import torch
import numpy as np

from torch.utils.data import Dataset
from dataset.fusion_utils import Voxelizer
from dataset.augmentation import ElasticDistortion, RandomHorizontalFlip, Compose
from utils.dataset_utils import load_dynamic_gaussian_npz


class DynamicFeatureDataset(Dataset):
    # Augmentation arguments
    SCALE_AUGMENTATION_BOUND = (0.9, 1.1)
    ROTATION_AUGMENTATION_BOUND = (
        (-np.pi / 64, np.pi / 64),
        (-np.pi / 64, np.pi / 64),
        (-np.pi, np.pi),
    )
    TRANSLATION_AUGMENTATION_RATIO_BOUND = ((-0.2, 0.2), (-0.2, 0.2), (0, 0))
    ELASTIC_DISTORT_PARAMS = ((0.2, 0.4), (0.8, 1.6))

    ROTATION_AXIS = "z"

    def __init__(self, gaussians_dir, point_dir, voxel_size=0.02, aug=False, feature_rotation=True):
        self.aug = aug
        self.scenes = os.listdir(point_dir)
        self.scenes.sort()

        self.data = []
        for scene in self.scenes:
            timesteps = os.listdir(os.path.join(point_dir, scene))
            timesteps.sort(key=lambda x: int(x))
            for t in timesteps:
                features = os.listdir(os.path.join(point_dir, scene, t))
                features.sort()
                for feature in features:
                    npz_path = os.path.join(gaussians_dir, scene, "params.npz")
                    feature_path = os.path.join(point_dir, scene, t, feature)
                    self.data.append([npz_path, feature_path, scene, t])

        self.voxelizer = Voxelizer(
            voxel_size=voxel_size,
            clip_bound=None,
            use_augmentation=aug,
            scale_augmentation_bound=self.SCALE_AUGMENTATION_BOUND,
            rotation_augmentation_bound=self.ROTATION_AUGMENTATION_BOUND,
            translation_augmentation_ratio_bound=self.TRANSLATION_AUGMENTATION_RATIO_BOUND,
        )

        self.prevoxel_transforms = Compose([ElasticDistortion(self.ELASTIC_DISTORT_PARAMS)])
        self.input_transforms = Compose([RandomHorizontalFlip(self.ROTATION_AXIS, is_temporal=False)])

    def __getitem__(self, index):
        with torch.no_grad():
            npz_path, feature_path, scene, t = self.data[index]
            locs, features = load_dynamic_gaussian_npz(npz_path, int(t))
            gt = torch.load(feature_path)
            features_gt, mask_chunk = gt["feat"], gt["mask_full"]

            # numpy transforms
            if self.aug:
                locs = self.prevoxel_transforms(locs)

            locs, features, _, inds_reconstruct, vox_ind = self.voxelizer.voxelize(
                locs, features, None, return_ind=True
            )

            vox_ind = torch.from_numpy(vox_ind)
            mask = mask_chunk[vox_ind]
            mask_ind = mask_chunk.nonzero(as_tuple=False)[:, 0]
            index1 = -torch.ones(mask_chunk.shape[0], dtype=int)
            index1[mask_ind] = mask_ind

            index1 = index1[vox_ind]
            chunk_ind = index1[index1 != -1]

            index2 = torch.zeros(mask_chunk.shape[0])
            index2[mask_ind] = 1
            index3 = torch.cumsum(index2, dim=0, dtype=int)

            indices = index3[chunk_ind] - 1
            features_gt = features_gt[indices]

            if self.aug:
                locs, features, _ = self.input_transforms(locs, features, None)

            locs = torch.from_numpy(locs).int()
            locs = torch.cat([torch.ones(locs.shape[0], 1, dtype=torch.int), locs], dim=1)
            features = torch.from_numpy(features).float()

        return locs, features, features_gt, mask

    def __len__(self):
        return len(self.data)
