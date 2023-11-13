import os
import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from dataset.dataset import GeoSegDataset, load_data_geo
from model import deeplabv3, unet, fcn
from model.deeplabv3plus import modeling
from utils.loss import FocalLoss, ohem_loss
from torch.utils.data import DataLoader


def get_argparser():
    parser = argparse.ArgumentParser()
    available_models = sorted(name for name in modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              modeling.__dict__[name])
                              )
    parser.add_argument("--model", type=str, default='deeplabv3plus_resnet101',
                        choices=available_models, help='model name')
    parser.add_argument("--data_dir", type=str, default="/data1/fyc/dataset/geo_seg",
                        help="path to Dataset")                    
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])
    parser.add_argument("--epoch", type=int, default=300,
                        help="epoch number (default: 300)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--batch_size", type=int, default=16,
                        help='batch size (default: 16)')
    parser.add_argument("--crop_size", type=int, default=512)
    parser.add_argument("--loss_type", type=str, default='cross_entropy',
                        choices=['cross_entropy', 'focal_loss'], help="loss type (default: False)")
    
    return parser


class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_utils`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):  #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


def main():
    opts = get_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: %s" % device)

    train_loader, val_loader = load_data_geo(opts.batch_size, (opts.crop_size, opts.crop_size), opts.data_dir)

    # model = net
    # model = getattr(deeplabv3, 'resnet50')(
    #     pretrained=True,
    #     num_classes=17
    # )
    # model = unet.UNet(3, n_classes=17)
    model = modeling.__dict__['deeplabv3plus_resnet101'](num_classes=17, 
                                                        output_stride=opts.output_stride)

    model.to(device)

    def CELoss(inputs, targets):
        return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

    FCLoss = FocalLoss(alpha=0.25, gamma=2)

    optimizer = Adam(model.parameters(), lr=opts.lr)

    best_acc = 0.

    for i in range(0, opts.epoch):
        print("===========第{}轮训练开始===========".format(i+1))
        train_metric = Accumulator(4)
        model.train()
        for idx, batch in enumerate(train_loader):
            data, targets = batch
            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data)
            if opts.loss_type == "cross_entropy":
                loss = CELoss(outputs, targets)
            else:
                loss = FCLoss(outputs, targets.squeeze(dim=1).long())
            train_acc = accuracy(outputs, targets)

            optimizer.zero_grad()
            loss.sum().backward()
            optimizer.step()

            train_metric.add(loss.sum(), train_acc, targets.shape[0], targets.numel())
            print(f'loss: {train_metric[0] / train_metric[2]:.3f}', 
                  f'train acc: {train_metric[1] / train_metric[3]:.3f}')


        model.eval()
        val_metric = Accumulator(2)
        with torch.no_grad():
            for idx, batch in enumerate(val_loader):
                data, targets = batch
                data = data.to(device)
                targets = targets.squeeze(dim=1).long().to(device)
                output = model(data)
                val_metric.add(accuracy(output, targets), targets.numel())
        val_acc = val_metric[0] / val_metric[1]
        
        print(f'val acc: {val_acc:.3f}')
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 
               os.path.join("./checkpoints", opts.model+"_weight.pth"))


if __name__ == "__main__":
    main()

