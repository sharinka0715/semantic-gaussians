import os
import sys

sys.path.append("./")
from scene.colmap_loader import read_points3D_binary, storePly

root = "../nerf/datasets/mvimgnet/"

# for root, dir, file in os.walk(root):
#     if "sparse" in dir:
#         print(root)
#         os.makedirs(root + "/sparse_aligned/0/", exist_ok=True)
#         os.system(
#             "colmap model_orientation_aligner --image_path {} --input_path {} --output_path {}".format(
#                 root + "/images", root + "/sparse/0", root + "/sparse_aligned/0/"
#             )
#         )


for root, dir, file in os.walk(root):
    if "sparse" in dir:
        # print(root)
        try:
            xyz, rgb, _ = read_points3D_binary(root + "/sparse/0/points3D.bin")
        except:
            print(root)
            continue
        ply_path = root + "/sparse/0/points3D.ply"
        storePly(ply_path, xyz, rgb)

        xyz, rgb, _ = read_points3D_binary(root + "/sparse_aligned/0/points3D.bin")
        ply_path = root + "/sparse_aligned/0/points3D.ply"
        storePly(ply_path, xyz, rgb)
