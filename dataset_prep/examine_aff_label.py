import os
from PIL import Image
from matplotlib import pyplot as plt

label = r"C:\ZTB\Code\ACGC-master\result\ir_label_vaihingen_0.55_0.05\top_mosaic_09cm_area1_0_2048_exist.png"
label = Image.open(label).convert('L')

plt.imshow(label)
plt.show()