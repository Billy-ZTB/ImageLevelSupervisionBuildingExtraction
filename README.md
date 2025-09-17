# Image Level Supervision for Remote Sensing Building Segmentation

## Usage

#### Step 1. 以VOC数据集风格整理数据集

- JPEGImages:根目录/JPEGImages 放置裁切好的影像数据，png格式；
- SegmentationClassAug:根目录/SegmentationClassAug 放置影像数据对应的语义分割标签，单通道，0为背景，255为前景；
- 生成cls_labels.npy:运行write_numpy.py文件，生成cls_labels.npy，放置于根目录下。


#### Step 2. 运行xxx.bat文件，开始训练

在.bat文件中编辑数据集根目录、权重输出路径等，一键运行，依次执行CAM训练、CAM生成、CAM评估、IRN label生成、IRN训练、伪标签生成步骤。

