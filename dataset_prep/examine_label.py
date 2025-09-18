import os
from PIL import Image
from matplotlib import pyplot as plt

label = r"C:\ZTB\Code\ISSS\result\sem_seg_vaihingen_0.5_0.4\top_mosaic_09cm_area1_0_2048_exist.png"
label = Image.open(label)

plt.imshow(label)
plt.show()