import os

root = r'C:\ZTB\Dataset\Vaihingen_wsss\test\image'
for img in os.listdir(root):
    print(img)
    name = img.split('.')[0]
    flag = name.split('_')[-1]
    if not flag=='expel':
        with open(f'C:\ZTB\Dataset\VOC_vaihingen/test.txt', 'a') as f:
            f.write(name+'\n')
