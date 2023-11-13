import torch
import torchfcn
from model.deeplabv3plus import modeling


model = modeling.__dict__['fcn32s'](num_classes=17, output_stride=8)
# model = torchfcn.models.FCN32s(n_class=17)
input = torch.randn(8, 3, 512, 512)
output = model(input)
print(output.shape)

