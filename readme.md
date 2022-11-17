# myresnet on Tiny-ImageNet dataset
## 实验背景问题
见`report.md`
## 环境配置
pytorch==1.12.1,torchvision==0.13.1
## 数据集准备
将`tiny_ImageNet_200_reorg`数据集解压后放在`data/`目录下

## 使用已训练好的模型evaluate
从 [onedrive分享](https://1drv.ms/u/s!Akd7I0kiaXxrgTrE8tOaproWQUtU?e=lJ2Beg) 下载已训练好的模型文件,将四个log文件夹赋值到项目路径下
### 使用ResNet
在`main.py`line 28, 设置变量`ISMYNET`的值`False`.
然后运行`python main.py ./data --evaluate --resume ./logs_resnet/`,
便可以使用保存的模型文件进行测试。
### 使用myResNet
在`main.py`line 28, 设置变量`ISMYNET`的值`True`.
然后运行`python main.py ./data --evaluate --resume ./logs_myresnet/model_best.pth.tar --p_dropout 0.5`.
通过修改参数为`--resume ./log_myresnet_dropout_0.3/model_best.pth.tar --p_dropout 0.3`或`--resume ./log_myresnet_dropout_0.7/model_best.pth.tar --p_dropout 0.7`来测试dropout概率为0.3或0.7时的模型

## train myresnet
在`main.py`line 28, 设置变量`ISMYNET`的值`True`.
然后运行`python main.py ./data --p_dropout [0-1]`.
训练完成后可以使用`python main.py ./data --evaluate --p_dropout [0-1]`来用训练得到的模型进行测试。
