import os

import cv2
import numpy as np


def count_building(label, txt_path):
    for _, _, filenames in os.walk(label):
        count1 = count2 = 0
        e = dict()
        for i in range(0, len(filenames)):
            print(filenames[i])
            png = os.path.join(label, filenames[i])  # png
            img = cv2.imread(png, -1)
            building = np.sum(img != 0)  # 统计建筑物像素
            if building >= 256 * 256 / 4:
                e[filenames[i].split('.')[0]] = [1,]
                count1 += 1
            elif building == 0:
                e[filenames[i].split('.')[0]] = [0,]
                count2 += 1
        np.save(txt_path, e)
        print(count1, count2)

label = 'C:\ZTB\Dataset\VOC_potsdam\SegmentationClassAug'
txt_path ='C:\ZTB\Dataset\VOC_potsdam/cls_labels.npy'
count_building(label, txt_path)