import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud, focal2fov, fov2focal
from utils.dataset_utils import SceneInfo, CameraInfo, getNerfppNorm, storePly, fetchPly


def readCamerasFromTransforms(path, transformsfile, white_background, extensions=[".png", ".jpg", ""]):
    cam_infos = []
    first_img = None

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            for extension in extensions:
                cam_name = os.path.join(path, frame["file_path"] + extension)
                if os.path.exists(cam_name):
                    break

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            if np.isinf(c2w).any():
                continue
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = Path(cam_name)
            image_name = image_path.stem
            if first_img is None:
                first_img = np.array(Image.open(image_path).convert("RGBA"))
                width, height = first_img.shape[1], first_img.shape[0]

            if "fl_x" in frame:
                fovx = focal2fov(frame["fl_x"], width)
                fovy = focal2fov(frame["fl_y"], height)
            else:
                fovx = contents["camera_angle_x"]
                fovy = focal2fov(fov2focal(fovx, width), height)

            if "intrinsics" in frame:
                intrinsics = np.array(frame["intrinsics"])
            else:
                intrinsics = np.zeros((4, 4), dtype=np.float32)
                intrinsics[0, 0] = fov2focal(fovx, width)
                intrinsics[1, 1] = fov2focal(fovy, height)
                intrinsics[2, 2] = 1
                intrinsics[3, 3] = 1
                intrinsics[0, 2] = width / 2
                intrinsics[1, 2] = height / 2

            cam_infos.append(
                CameraInfo(
                    uid=idx,
                    R=R,
                    T=T,
                    FovY=fovy,
                    FovX=fovx,
                    image_path=image_path,
                    image_name=image_name,
                    width=width,
                    height=height,
                    intrinsics=intrinsics,
                )
            )

    return cam_infos


def readBlenderInfo(path, white_background, eval, extensions=[".png", ".jpg", ""]):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extensions)
    print("Reading Test Transforms")
    try:
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extensions)
    except:
        print("Reading Test Transforms Error! Skip it.")
        test_cam_infos = []

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info
