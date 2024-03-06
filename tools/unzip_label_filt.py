import os
import zipfile
from tqdm import tqdm
import traceback

LABEL_ROOT = "/PATH/TO/YOUR/SCANNET/RAW"
OUT_ROOT = "/PATH/TO/YOUR/SCANNET/EXTRACTED"


for split in ["val"]:
    ls = os.listdir(os.path.join(OUT_ROOT, split))
    ls.sort()
    for scene in tqdm(ls):
        img_path = os.path.join(OUT_ROOT, split, scene, "color")
        ext_imgs = os.listdir(img_path)
        ext_imgs.sort()

        out_path = os.path.join(OUT_ROOT, split, scene)
        os.makedirs(out_path, exist_ok=True)
        label_zip = os.path.join(LABEL_ROOT, f"{scene}_2d-label-filt.zip")
        with zipfile.ZipFile(label_zip, "r") as zip_ref:
            for img in ext_imgs:
                try:
                    zip_ref.extract(f"label-filt/{img}".replace("jpg", "png"), out_path)
                except Exception:
                    print(traceback.format_exc())
                    print(scene)
