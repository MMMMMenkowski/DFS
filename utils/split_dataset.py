import os
import random
import numpy as np
import math
import shutil


origin_dir = r"D:\project\SOD\geology_seg\ori_dataset"
target_dir = r"D:\project\SOD\dataset\geo_seg"

# for file in os.listdir(os.path.join(origin_dir, "labels")):
label_list = os.listdir(os.path.join(origin_dir, "labels"))
ratio = 0.25
val_index = random.sample(label_list, k=math.ceil(len(label_list) * ratio))

for file in label_list:
    if file in val_index:
        shutil.copy(os.path.join(origin_dir, "labels", file),
                    os.path.join(target_dir, "val", "SegClasses", file))
        shutil.copy(os.path.join(origin_dir, "images", "pos", file.split(".")[0]+"+.jpg"),
                    os.path.join(target_dir, "val", "Images", "pos", file.split(".")[0]+"+.jpg"))
        shutil.copy(os.path.join(origin_dir, "images", "neg", file.split(".")[0] + "-.jpg"),
                    os.path.join(target_dir, "val", "Images", "neg", file.split(".")[0] + "-.jpg"))

    else:
        shutil.copy(os.path.join(origin_dir, "labels", file),
                    os.path.join(target_dir, "train", "SegClasses", file))
        shutil.copy(os.path.join(origin_dir, "images", "pos", file.split(".")[0]+"+.jpg"),
                    os.path.join(target_dir, "train", "Images", "pos", file.split(".")[0]+"+.jpg"))
        shutil.copy(os.path.join(origin_dir, "images", "neg", file.split(".")[0] + "-.jpg"),
                    os.path.join(target_dir, "train", "Images", "neg", file.split(".")[0] + "-.jpg"))