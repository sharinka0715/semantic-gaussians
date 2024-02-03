import os
import json
import argparse

parser = argparse.ArgumentParser()
# data paths
parser.add_argument(
    "--input_path",
    required=True,
    help="path to input folder (e.g. ../scannet/train/)",
)
opt = parser.parse_args()

scene = opt.input_path

os.system(f"mv {scene}/color {scene}/images")
intrinsics = {
    "fl_x": 1170.187988 * 320 / 1296,
    "fl_y": 1170.187988 * 240 / 968,
    "cx": 159.750000,
    "cy": 119.750000,
}
os.system(
    f"colmap feature_extractor --database_path {scene}/database.db --image_path {scene}/images --ImageReader.camera_model PINHOLE --ImageReader.camera_params {intrinsics['fl_x']},{intrinsics['fl_y']},{intrinsics['cx']},{intrinsics['cy']}"
)
os.system(f"colmap exhaustive_matcher --database_path {scene}/database.db")
os.makedirs(os.path.join(scene, "sparse"), exist_ok=True)

os.system(f"colmap mapper --database_path {scene}/database.db --image_path {scene}/images --output_path {scene}/sparse")

