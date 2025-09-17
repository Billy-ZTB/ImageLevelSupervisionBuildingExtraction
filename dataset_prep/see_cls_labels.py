import numpy as np

npy = np.load("C:\ZTB\Dataset\VOC_potsdam\cls_labels.npy", allow_pickle=True).item()
print(npy['top_potsdam_2_10_0_1024_exist'])
#print(npy)