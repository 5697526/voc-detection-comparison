# 在VOC数据集上训练并测试模型 Mask R-CNN 和 Sparse R-CNN 

### 一、项目概述：
使用现成的目标检测框架[mmdetection](https://github.com/open-mmlab/mmdetection) 在 VOC 数据集上训练并测试目标检测模型 Mask R-CNN 和 Sparse R-CNN。对训练好的模型进行可视化对比，包括 Mask R-CNN 第一阶段的 proposal box 和最终预测结果，以及 Mask R-CNN 和 Sparse R-CNN 的实例分割与目标检测结果。此外，还会使用不在 VOC 数据集内但包含 VOC 类别物体的图像进行测试。

### 二、数据集介绍
PASCAL VOC 是计算机视觉领域最经典的目标检测和图像分割基准数据集之一，其中VOC2007和VOC2012是最常用的两个版本，共有20个常见物体类别（如人、车辆、动物等）。数据集提供约1.7万张标注图像，每张图片包含XML格式的物体边界框标注和分割掩码，支持目标检测、分类和分割任务。其特点在于中等规模、场景多样、标注精细，常被用作算法性能测试的标准基准，评估指标主要采用mAP（平均精度均值）。


### 三、文件结构

`
voc-detection-comparison/mmdetection/
├── configs/    #存放模型训练和测试的配置文件
│   ├── mask-rcnn_voc.py
│   └── sparse-rcnn_voc.py
├── demo/       #存放示例代码或演示相关的文件
├── data/       #数据集相关目录
│   └── VOCdevkit/
│       ├── VOC2007/
│       └── VOC2012/
├── mmdet/      # mmdetection 相关文件
├── results/    #存放可视化结果
├── tools/      #包含训练和测试的脚本
│   ├── test.py
│   └── train.py
└── work_dirs/  #训练过程中的模型权重、日志等文件会保存在此
`


### 四、训练和测试步骤
#### 1. 安装依赖
`
conda create -n mmdet python=3.8 -y
conda activate mmdet
pip install torch torchvision torchaudio
pip install -U openmim
mim install "mmcv<2.2.0" 
cd mmdetection
pip install -v -e .
`

#### 2.  数据集准备

`
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
`

#### 3.  训练模型
`

`

`

`

### 五、模型权重下载
