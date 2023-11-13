import os
import base64
import json
import torch
import torchvision
import argparse
from io import BytesIO
from tqdm import tqdm
from PIL import Image
from glob import glob
from model import deeplabv3
from model.deeplabv3plus import modeling
from dataset.dataset import rand_crop, json_to_label, label_dic


def get_argparser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, required=True,
                        help="path to a single image or image directory")
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        help='model name')
    parser.add_argument("--weights_path", type=str, default='./checkpoints',
                        help="path to load model weights")
    parser.add_argument("--output_stride", type=int, default=8, choices=[8, 16])
    parser.add_argument("--num_classes", type=int, default=17)
    parser.add_argument("--save_results_to", type=str, default='./save_images',
                        help="path to save_results")

    return parser


def predict(img_path, device, net, pos=True):
    PILToTensor = torchvision.transforms.PILToTensor()
    img = PILToTensor(Image.open(img_path))
    transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    X = transform(img.float() / 255.)
    pred = net(X.unsqueeze(0).to(device)).argmax(dim=1)

    return pred


def decode_and_save(pred, save_path):
    img_cls = pred.squeeze(0)
    h, w = img_cls.shape
    img = torch.zeros(h, w)

    labels = []
    for key, value in label_dic.items():
        if value in img_cls:
            labels.append(key)
    labels.sort(key=lambda x: x in labels)
    for label in labels:
        img[img_cls==label_dic[label]] = labels.index(label)
    
    img = Image.fromarray(img.numpy())
    # img.save("."+save_path.split(".")[1]+".png")
    img_buffer = BytesIO()
    img.save(img_buffer, format="png")
    img_data = base64.b64encode(img_buffer.getvalue())
    
    dic = dict()
    dic["labels"] = labels
    dic["image_data"] = img_data.decode()
    json.dump(dic, open(save_path, 'w'), indent=4)



def main():
    opts = get_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    # 加载数据
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s'%(ext)), recursive=True)
            if len(files)>0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)
    

    # 加载模型
    model = modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    param_path = os.path.join(opts.weights_path, opts.model+"_weight.pth")
    model.load_state_dict(torch.load(param_path))
    model.to(device)

    if opts.save_results_to is not None:
        os.makedirs(opts.save_results_to, exist_ok=True)
    with torch.no_grad():
        model = model.eval()
        for img_path in image_files:
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext)-1]
            pred = predict(img_path, device, model)
            decode_and_save(pred, os.path.join(opts.save_results_to, img_name+".json"))



main()
            

