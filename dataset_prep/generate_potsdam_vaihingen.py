import os
import random
import rasterio
from PIL import Image
import numpy as np
import glob
random.seed(42)

def crop_image_with_overlap(image, crop_size=256, overlap_ratio=0.5):
    """
    对图像进行滑动窗口裁剪
    Args:
        image: PIL Image对象
        crop_size: 裁剪大小 (默认256)
        overlap_ratio: 重叠比例 (默认0.5)
    Returns:
        crops: 裁剪后的图像块列表
    """
    image_array = np.array(image)
    h, w = image_array.shape[:2]

    # 计算步长
    step = int(crop_size * (1 - overlap_ratio))

    crops = []
    crop_positions = []

    # 滑动窗口裁剪
    for y in range(0, h - crop_size + 1, step):
        for x in range(0, w - crop_size + 1, step):
            # 裁剪图像块
            if len(image_array.shape) == 3:  # RGB图像
                crop = image_array[y:y + crop_size, x:x + crop_size, :]
            else:  # 灰度图像
                crop = image_array[y:y + crop_size, x:x + crop_size]

            crops.append(Image.fromarray(crop))
            crop_positions.append((x, y))

    return crops, crop_positions


train_and_valid_labels = r'C:\ZTB\Dataset\postdam\5_Labels_for_participants\5_Labels_for_participants'
all_labels = r'C:\ZTB\Dataset\postdam\5_Labels_all'
rgb = r'C:\ZTB\Dataset\postdam\4_Ortho_RGBIR\4_Ortho_RGBIR'  # RGB图像文件夹路径
dataset_output = r'C:\ZTB\Dataset\Potsdam_wsss'

# 获取train_and_valid_labels文件夹下的所有tif文件
train_valid_tif_files = []
if os.path.exists(train_and_valid_labels):
    train_valid_tif_files = glob.glob(os.path.join(train_and_valid_labels, "*.tif"))
    #train_valid_tif_files.extend(glob.glob(os.path.join(train_and_valid_labels, "*.TIF")))

# 获取all_labels文件夹下的所有tif文件
all_tif_files = []
if os.path.exists(all_labels):
    all_tif_files = glob.glob(os.path.join(all_labels, "*.tif"))
    #all_tif_files.extend(glob.glob(os.path.join(all_labels, "*.TIF")))

# 提取文件名（不包含路径）用于比较
train_valid_filenames = [os.path.basename(f) for f in train_valid_tif_files]
all_filenames = [os.path.basename(f) for f in all_tif_files]

# 找出在all_labels里但不在train_and_valid_labels里的文件
test_only_filenames = list(set(all_filenames) - set(train_valid_filenames))
test_only_files = [os.path.join(all_labels, filename) for filename in test_only_filenames]

for file in all_tif_files:
    split = ''
    if file not in test_only_files:
        print('Processing train/valid set file:', file)

    else:
        print('Processing test set file:', file)
        split = 'test'
    i = 0

    filename_without_ext = os.path.splitext(os.path.basename(file))[0]
    filename = filename_without_ext[:filename_without_ext.rfind('_')]
    if filename.startswith('top_potsdam_7_10'):
        continue
    with rasterio.open(os.path.join(rgb, f'{filename}_RGBIR.tif')) as src:
        # 读取所有波段
        img_array = src.read()  # 形状为 (bands, height, width)
        # 转换为正确的格式 (height, width, channels)
        rgb_array = np.transpose(img_array[:3, :, :], (1, 2, 0))
        rgb_image = Image.fromarray(rgb_array.astype(np.uint8))

        label = Image.open(file)
        label_array = np.array(label)
        building_array = np.all(label_array == [0, 0, 255], axis=2)
        building_image = Image.fromarray(building_array.astype(np.uint8) * 255)

        # 对RGB图像和建筑物掩码进行对应裁剪
        rgb_crops, crop_positions = crop_image_with_overlap(rgb_image, crop_size=256, overlap_ratio=0.5)
        building_crops, _ = crop_image_with_overlap(building_image, crop_size=256, overlap_ratio=0.5)

        # 保存裁剪后的图像块
        for idx, (rgb_crop, building_crop, pos) in enumerate(zip(rgb_crops, building_crops, crop_positions)):
            np_building_crop = np.array(building_crop)
            building_flag = ''
            if np.sum(np_building_crop==255) == 0:
                building_flag = 'non'
            elif np.sum(np_building_crop==255) >= 256*256/4:
                building_flag = 'exist'
            else:
                building_flag = 'expel'
            x, y = pos
            crop_filename = f"{filename}_{x}_{y}_{building_flag}.png"
            if not split=='test':
                is_valid = random.random() < 1 / 9
                if is_valid:
                    split = 'valid'
                else:
                    split = 'train'
            print('File ', crop_filename,' is split to ', split, ' set')
            image_output = os.path.join(dataset_output, split, 'image')
            label_output = os.path.join(dataset_output, split, 'label')
            os.makedirs(image_output, exist_ok=True)
            os.makedirs(label_output, exist_ok=True)
            # 保存裁剪后的图像
            rgb_crop.save(os.path.join(image_output, crop_filename))
            building_crop.save(os.path.join(label_output,crop_filename))

        print(f"处理完成 {filename}: 生成了 {len(rgb_crops)} 个裁剪块")
