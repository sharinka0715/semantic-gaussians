import os
import shutil

root = "./output/scannet"

train = os.path.join(root, "train")
val = os.path.join(root, "val")

with open("./dataset/scannet/scannetv2_train.txt") as fp:
    data = [e.strip() for e in fp.readlines()]

os.makedirs(train, exist_ok=True)
for scene in data:
    shutil.move(os.path.join(root, scene), train)

with open("./dataset/scannet/scannetv2_val.txt") as fp:
    data = [e.strip() for e in fp.readlines()]

os.makedirs(val, exist_ok=True)
for scene in data:
    shutil.move(os.path.join(root, scene), val)
