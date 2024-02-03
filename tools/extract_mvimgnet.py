import os
import zipfile
from tqdm import tqdm
import random

file_info = {}
RANDOM_FILE_NUM = 20

ls = os.listdir("/home/guojun/fillipo/Datasets/MVImgNet/data/")
ls.sort()


def unzip_directory(zip_path, target_directory, start):
    with zipfile.ZipFile(zip_path, "r") as fp:
        all_files = fp.namelist()
        target_files = [file for file in all_files if file.startswith(start)]
        for file in target_files:
            fp.extract(file, target_directory)


for zips in tqdm(ls):
    fp = zipfile.ZipFile(f"/home/guojun/fillipo/Datasets/MVImgNet/data/{zips}")
    for file in fp.namelist():
        file = file.split("/")
        if len(file) < 2:
            continue
        cl = int(file[0])
        idx = file[1]
        pl = zips + "/" + idx
        if cl not in file_info:
            file_info[cl] = []
        if pl not in file_info[cl]:
            file_info[cl].append(pl)

for key in [0, 1, 5, 6, 23, 24, 28, 35, 36, 37, 42, 44, 45, 48, 50, 85, 103, 104, 111, 156]:
    ls = file_info[key]
    # If samples in this class is larger than RANDOM_FILE_NUM, random samping from them.
    if len(file_info[key]) > RANDOM_FILE_NUM:
        ls = random.sample(file_info[key], RANDOM_FILE_NUM)

    for z in ls:
        zips, idx = z.split("/")
        unzip_directory(
            f"/home/guojun/fillipo/Datasets/MVImgNet/data/{zips}",
            f"/home/guojun/winshare/nerf/datasets/mvimgnet/",
            f"{key}/{idx}",
        )
