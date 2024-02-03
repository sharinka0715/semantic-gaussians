import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud, focal2fov, fov2focal
from utils.dataset_utils import SceneInfo, CameraInfo, getNerfppNorm, storePly, fetchPly


def readReplicaInfo(path, white_background, eval, extensions=[".png", ".jpg"]):
    path = Path(path)
    image_dir = path / "rgb"
    image_sorted = list(sorted(image_dir.iterdir(), key=lambda x: int(x.name.split(".")[0].split("_")[1])))
    poses = np.loadtxt(path / "traj_w_c.txt").reshape(-1, 4, 4).astype(np.float32)  # [N,16] -> [N,4,4]
    train_cam_infos = []

    first_img = np.array(Image.open(image_sorted[0]).convert("RGBA"))
    width, height = first_img.shape[1], first_img.shape[0]

    fovx = np.radians(90)
    fovy = focal2fov(fov2focal(fovx, width), height)
    K = np.zeros((4, 4), dtype=np.float32)

    K[0, 0] = fov2focal(fovx, width)
    K[1, 1] = fov2focal(fovy, height)
    K[2, 2] = 1
    K[3, 3] = 1
    K[0, 2] = width / 2
    K[1, 2] = height / 2

    trans = []
    for idx, img in enumerate(image_sorted):
        c2w = poses[idx]
        trans.append(c2w[:3, 3])

        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = img
        image_name = Path(img).stem

        train_cam_infos.append(
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
                intrinsics=K,
            )
        )
    center = np.array(trans).mean(axis=0, keepdims=True)
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = (np.random.random((num_pts, 3)) - 0.5) * 2
        xyz = (xyz + center) * 1.5
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
        test_cameras=[],
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info
