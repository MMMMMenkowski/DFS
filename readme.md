# 砂砾图像分割

## 1.可供训练的模型

目前包括fcn、UNet、deeplabv3和deeplabv3+，由于fcn和UNet性能较差，暂不考虑使用。deeplabv3和deeplabv3+都定义在model.deeplabv3plus下。

## 2.训练
确保安装requirements.txt中的库后，运行train.sh。训练完后的模型参数会以pth格式保存在checkpoints中

## 3.测试
运行pred.sh以测试指定文件夹中的图片，会在save_images文件夹中以json格式保存生成的图片。