import os
import json
import base64
import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from io import BytesIO
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


label_dic = {"_background_": 0,
             "Cht": 1,
             "Lmp": 2,
             "Lms": 3,
             "Lsc": 4,
             "Lss": 5,
             "Others": 6,
             "Qm": 7,
             "Iii": 8,
             "Lsm": 9,
             "P": 10,
             "P1": 11,
             "Qp": 12,
             "K": 13,
             "Lim": 14,
             "Lvf": 15,
             "Lvm": 16}


def json_to_label(json_path):
    """将json文件转换为标签图像"""
    with open(json_path) as f:
        dic = json.load(f)
        labels = dic["labels"]
        image_data = dic["image_data"]
    label_to_class = {}
    for i, label_name in enumerate(labels):
        label_to_class[i] = label_dic[label_name.split(":")[0].strip()]
    # print(label_to_class)
    image = base64.b64decode(image_data)
    img = Image.open(BytesIO(image))
    # img.show()
    PILToTensor = transforms.PILToTensor()
    img = PILToTensor(img).to(torch.int64).squeeze(dim=0)
    h, w =img.shape
    img_cls = torch.zeros(h, w)
    for key, value in label_to_class.items():
        img_cls[img == key] = value
    return img_cls.unsqueeze(dim=0)


# json_path = r"/data1/fyc/dataset/geo_seg"
# img = json_to_label(json_path)
# print(img[105:115, 130:140])


def read_images(img_dir, mode="pos", is_train=True):
    """读取所有数据集图像并标注"""
    features, labels = [], []
    file_pos_dir = os.path.join(img_dir, "train" if is_train else "val", "Images", "pos")
    file_neg_dir = os.path.join(img_dir, "train" if is_train else "val", "Images", "neg")
    label_dir = os.path.join(img_dir, "train" if is_train else "val", "SegClasses")
    file_list, label_list = [], []
    if mode == "both":
        for root, _, file in os.walk(label_dir):
            for name in file:
                label_list.append(os.path.join(root, name))
                label_list.append(os.path.join(root, name))
                file_list.append(os.path.join(file_pos_dir, name.split(".")[0] + "+.jpg"))
                file_list.append(os.path.join(file_neg_dir, name.split(".")[0] + "-.jpg"))
    elif mode == "pos":
        for root, _, file in os.walk(label_dir):
            for name in file:
                label_list.append(os.path.join(root, name))
                file_list.append(os.path.join(file_pos_dir, name.split(".")[0] + "+.jpg"))
    elif mode == "neg":
        for root, _, file in os.walk(label_dir):
            for name in file:
                label_list.append(os.path.join(root, name))
                file_list.append(os.path.join(file_neg_dir, name.split(".")[0] + "-.jpg"))

    assert len(file_list) == len(label_list)

    PILToTensor = transforms.PILToTensor()

    for i in range(len(file_list)):
        features.append(PILToTensor(Image.open(file_list[i])))
        labels.append(json_to_label(label_list[i]))

    return features, labels


# img_dir = "/data1/fyc/dataset/geo_seg"
# f, l = read_images(img_dir)
# print(f[0].shape, l[0].shape)


def rand_crop(feature, label, height, width):
    """随机裁剪特征和标签图像"""
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width))
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label


# f_c, l_c = rand_crop(f[0], l[0], 2000, 3000)
# plt.subplot(121)
# plt.imshow(f_c.permute(1,2,0))
# plt.subplot(122)
# plt.imshow(l_c)
# plt.show()


class GeoSegDataset(Dataset):
    def __init__(self, is_train, crop_size, data_dir, mode):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_images(data_dir, mode=mode, is_train=is_train)
        # print(features[0].shape, labels[0].shape)
        self.features = [self.normalize_image(feature)
                         for feature in self.filter(features)]
        self.labels = self.filter(labels)
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255.)

    def filter(self, imgs):
        return [img for img in imgs if (
                img.shape[1] >= self.crop_size[0] and
                img.shape[2] >= self.crop_size[1])]

    def __getitem__(self, idx):
        feature, label = rand_crop(self.features[idx], self.labels[idx],
                                   *self.crop_size)
        return feature, label

    def __len__(self):
        return len(self.features)


def load_data_geo(batch_size, crop_size, data_dir):
    """加载地质语义分割数据集"""
    mode = "neg"
    num_workers = 4
    train_iter = torch.utils.data.DataLoader(
        GeoSegDataset(True, crop_size, mode=mode, data_dir=data_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    val_iter = torch.utils.data.DataLoader(
        GeoSegDataset(False, crop_size, mode=mode,data_dir=data_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter, val_iter


if __name__ == "__main__":
    pass