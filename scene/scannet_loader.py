import os
import json
import numpy as np
from PIL import Image
from pathlib import Path

from utils.sh_utils import SH2RGB
from utils.graphics_utils import BasicPointCloud, focal2fov, fov2focal
from utils.dataset_utils import SceneInfo, CameraInfo, getNerfppNorm, storePly, fetchPly


def readScanNetInfo(path, white_background, eval, llffhold=8, extensions=[".png", ".jpg"]):
    path = Path(path)
    image_dir = path / "color"
    pose_dir = path / "pose"
    image_sorted = list(sorted(image_dir.iterdir(), key=lambda x: int(x.name.split(".")[0])))
    pose_sorted = list(sorted(pose_dir.iterdir(), key=lambda x: int(x.name.split(".")[0])))

    cam_infos = []
    K = np.loadtxt(os.path.join(path, "intrinsic/intrinsic_color.txt"))
    first_img = np.array(Image.open(image_sorted[0]).convert("RGBA"))
    width, height = first_img.shape[1], first_img.shape[0]

    fovx = focal2fov(K[0, 0], K[0, 2] * 2)
    fovy = focal2fov(K[1, 1], K[1, 2] * 2)

    i = 0
    for img, pose in zip(image_sorted, pose_sorted):
        i += 1
        idx = int(img.name.split(".")[0])
        c2w = np.loadtxt(pose)
        c2w = np.array(c2w).reshape(4, 4).astype(np.float32)
        # ScanNet pose use COLMAP coordinates (Y down, Z forward), so no need to flip the axis
        # c2w[:3, 1:3] *= -1
        # We cannot accept files directly, as some of the poses are invalid
        if np.isinf(c2w).any():
            continue

        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image_path = img
        image_name = Path(img).stem

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
                intrinsics=K,
            )
        )

    nerf_normalization = getNerfppNorm(cam_infos)

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

    pcd = fetchPly(ply_path)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    scene_info = SceneInfo(
        point_cloud=pcd,
        train_cameras=train_cam_infos,
        test_cameras=test_cam_infos,
        nerf_normalization=nerf_normalization,
        ply_path=ply_path,
    )
    return scene_info
