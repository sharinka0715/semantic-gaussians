import os
import json
import random
from torch.utils.data import Dataset
from scene.blender_loader import readBlenderInfo
from scene.colmap_loader import readColmapInfo
from scene.scannet_loader import readScanNetInfo
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON, loadCam

sceneLoadTypeCallbacks = {
    "Colmap": readColmapInfo,
    "Blender": readBlenderInfo,
    "ScanNet": readScanNetInfo,
}


class SceneDataset(Dataset):
    def __init__(self, scene_info, args, split="train", resolution_scale=1.0):
        self.scene_info = scene_info
        self.args = args
        self.resolution_scale = resolution_scale
        if split == "train":
            self.camera_info = scene_info.train_cameras
        elif split == "test":
            self.camera_info = scene_info.test_cameras
        else:
            raise NotImplementedError("Undefined split")

    def __len__(self):
        return len(self.camera_info)

    def __getitem__(self, index):
        return loadCam(self.args, index, self.camera_info[index], self.resolution_scale)


class Scene:
    def __init__(self, args, resolution_scales=[1.0]):
        self.train_cameras = {}
        self.test_cameras = {}

        # load scene
        if os.path.exists(os.path.join(args.scene_path, "pose")):
            print("Found pose directory, assuming ScanNet data set!")
            scene_info = sceneLoadTypeCallbacks["ScanNet"](
                args.scene_path,
                args.colmap_images,
                args.test_cameras,
                args.colmap_eval_hold,
            )
        elif os.path.exists(os.path.join(args.scene_path, "sparse")):
            print("Found sparse directory, assuming Colmap data set!")
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.scene_path,
                args.colmap_images,
                args.test_cameras,
                args.colmap_eval_hold,
            )
        elif os.path.exists(os.path.join(args.scene_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender synthetic data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.scene_path, args.white_background, args.test_cameras)
        # elif os.path.exists(os.path.join(args.scene_path, "traj_w_c.txt")):
        #     print("Found traj_w_c.txt file, assuming Replica data set!")
        #     scene_info = sceneLoadTypeCallbacks["Replica"](args.scene_path, args.white_background, args.test_cameras)
        else:
            assert False, "Could not recognize scene type!"

        self.pcd = scene_info.point_cloud
        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras...")
            self.train_cameras[resolution_scale] = SceneDataset(scene_info, args, "train", resolution_scale)
            print("Loading Test Cameras...")
            self.test_cameras[resolution_scale] = SceneDataset(scene_info, args, "test", resolution_scale)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
