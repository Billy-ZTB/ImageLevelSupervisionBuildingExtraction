import os
import cv2
import glob
import torch

import math
import imageio
import numpy as np

from PIL import Image

from core.aff_utils import *

from tools.ai.augment_utils import *
from tools.ai.torch_utils import one_hot_embedding

from tools.general.xml_utils import read_xml
from tools.general.json_utils import read_json
from tools.dataset.voc_utils import get_color_map_dic

class Iterator:
    def __init__(self, loader):
        self.loader = loader
        self.init()

    def init(self):
        self.iterator = iter(self.loader)
    
    def get(self):
        try:
            data = next(self.iterator)
        except StopIteration:
            self.init()
            data = next(self.iterator)
        
        return data

class VOC_Dataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, domain, with_id=False, with_tags=False, with_mask=False):
        self.root_dir = root_dir

        self.image_dir = self.root_dir + 'JPEGImages/'
        self.xml_dir = self.root_dir + 'Annotations/'
        self.mask_dir = self.root_dir + 'SegmentationClass/'##
        # self.mask_dir = self.root_dir + 'SegmentationClass/'
        self.image_id_list = [image_id.strip() for image_id in open(r'D:\Experiments\Potsdam\data\train\{}.txt'.format(domain)).readlines()]
        
        self.with_id = with_id
        self.with_tags = with_tags
        self.with_mask = with_mask

    def __len__(self):
        return len(self.image_id_list)

    def get_image(self, image_id):
        # image = Image.open(self.image_dir + image_id + '.jpg').convert('RGB')
        image = imageio.imread(self.image_dir + image_id + '.tif')[:,:,:3]
        # image = imageio.imread(self.image_dir + image_id + '.tif')##
        return image

    def get_mask(self, image_id):
        mask_path = self.mask_dir + image_id + '.png'
        if os.path.isfile(mask_path):
            # mask = cv2.imread(mask_path)[:,:,0] / 255
            mask = cv2.imread(mask_path)[:, :, 0]
        else:
            mask = None
        return mask

    def get_tags(self, image_id):
        # _, tags = read_xml(self.xml_dir + image_id + '.xml')
        d = np.load(r'D:\Experiments\Potsdam\data\train\cls_labels.npy', allow_pickle=True).item()
        # tags = d[image_id][1:]
        tags = d[image_id]
        return tags
    
    def __getitem__(self, index):
        image_id = self.image_id_list[index]

        data_list = [self.get_image(image_id)]

        if self.with_id:
            data_list.append(image_id)

        if self.with_tags:
            data_list.append(self.get_tags(image_id))

        if self.with_mask:
            data_list.append(self.get_mask(image_id))
        
        return data_list

class VOC_Dataset_For_Classification(VOC_Dataset):
    def __init__(self, root_dir, domain, transform=None):
        super().__init__(root_dir, domain, with_tags=True)
        self.transform = transform

        # cmap_dic, _, class_names = get_color_map_dic()
        # cmap_dic = {'background': [255, 0, 0],
        #             'impervious_surfaces':[255, 255, 255],
        #             'building': [0, 0, 255],
        #             'low_vegetation': [0, 255, 255],
        #             'tree': [0, 255, 0],
        #             'car': [255, 255, 0]}
        # self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])


        # self.class_dic = {'background':0,'car':1}
        # self.classes = 2

    def __getitem__(self, index):
        image, tags = super().__getitem__(index)

        if self.transform is not None:
            image = self.transform(image)

        label = tags
        # label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
        return image, label

class VOC_Dataset_For_Segmentation(VOC_Dataset):
    def __init__(self, root_dir, domain, transform=None):
        super().__init__(root_dir, domain, with_mask=True)
        self.transform = transform

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])
    
    def __getitem__(self, index):
        image, mask = super().__getitem__(index)

        if self.transform is not None:
            input_dic = {'image':image, 'mask':mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']
            # mask = output_dic['mask']
            mask = (output_dic['mask'] / 255).astype('int64')
        return image, mask

class VOC_Dataset_For_Evaluation(VOC_Dataset):
    def __init__(self, root_dir, domain, transform=None):
        super().__init__(root_dir, domain, with_id=True, with_mask=True)
        self.transform = transform

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])

    def __getitem__(self, index):
        image, image_id, mask = super().__getitem__(index)

        if self.transform is not None:
            input_dic = {'image':image, 'mask':mask}
            output_dic = self.transform(input_dic)

            # image = output_dic['image']
            # # mask = output_dic['mask']
            # mask = (output_dic['mask'] / 255)
        return image, image_id, mask

class VOC_Dataset_For_WSSS(VOC_Dataset):
    def __init__(self, root_dir, domain, pred_dir, transform=None):
        super().__init__(root_dir, domain, with_id=True)
        self.pred_dir = pred_dir
        self.transform = transform

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])
    
    def __getitem__(self, index):
        image, image_id = super().__getitem__(index)
        mask = cv2.imread(self.pred_dir + image_id + '.png')[:,:,0] ##
        # mask = cv2.imread(self.pred_dir + image_id + '.png')

        if self.transform is not None:
            input_dic = {'image':image, 'mask':mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']
            # mask = output_dic['mask']
            mask = (output_dic['mask'] / 255).astype('int64')
        
        return image, mask

class VOC_Dataset_For_Testing_CAM(VOC_Dataset):
    def __init__(self, root_dir, domain, transform=None):
        super().__init__(root_dir, domain, with_tags=True, with_mask=True)
        self.transform = transform

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])
        
        data = read_json('./data/VOC_2012.json')

        self.class_dic = data['class_dic']
        self.classes = data['classes']

        # self.class_dic = {'background': 0, 'car': 1}
        # self.classes = 2

    def __getitem__(self, index):
        image, tags, mask = super().__getitem__(index)

        if self.transform is not None:
            input_dic = {'image':image, 'mask':mask}
            output_dic = self.transform(input_dic)

            image = output_dic['image']
            mask = (output_dic['mask'] / 255).astype('int64') ##
            # mask = output_dic['mask']

        label = tags
        # label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
        return image, label, mask

class VOC_Dataset_For_Making_CAM(VOC_Dataset):
    def __init__(self, root_dir, domain):
        super().__init__(root_dir, domain, with_id=True, with_tags=True, with_mask=True)

        cmap_dic, _, class_names = get_color_map_dic()
        self.colors = np.asarray([cmap_dic[class_name] for class_name in class_names])
        
        data = read_json('./data/VOC_2012.json')
        #
        # self.class_dic = {'background': 0, 'car': 1}
        # self.classes = 2
        # self.class_names = ['background', 'car']
        self.class_names = np.asarray(class_names[1:6])
        self.class_dic = data['class_dic']
        self.classes = data['classes']

    def __getitem__(self, index):
        image, image_id, tags, mask = super().__getitem__(index)
        label = tags
        # label = one_hot_embedding([self.class_dic[tag] for tag in tags], self.classes)
        return image, image_id, label, mask

class VOC_Dataset_For_Affinity(VOC_Dataset):
    def __init__(self, root_dir, domain, path_index, label_dir, transform=None):
        super().__init__(root_dir, domain, with_id=True)

        data = read_json('./data/VOC_2012.json')

        self.class_dic = data['class_dic']
        self.classes = data['classes']

        # self.class_dic = {'background': 0, 'car': 1}
        # self.classes = 2

        self.transform = transform

        self.label_dir = label_dir
        self.path_index = path_index

        self.extract_aff_lab_func = GetAffinityLabelFromIndices(self.path_index.src_indices, self.path_index.dst_indices)

    def __getitem__(self, idx):
        image, image_id = super().__getitem__(idx)

        label = imageio.imread(self.label_dir + image_id + '.png')
        label = Image.fromarray(label)
        
        output_dic = self.transform({'image':image, 'mask':label})
        image, label = output_dic['image'], output_dic['mask']
        
        return image, self.extract_aff_lab_func(label)


import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm as tqdm
from PIL import Image
from torchvision.transforms.functional import to_tensor


def read_image_names(path, step, folder='image'):
    assert step in ['train', 'valid', 'test']
    path = os.path.join(path, step, folder)  #if step in ['train', 'valid'] else os.path.join(path, step, 'image')
    names = []
    for idx, file in enumerate(tqdm(os.listdir(path))):
        name = os.path.splitext(file)[0]
        names.append(name)
        '''if idx > 100:
            return names'''
    return names

suffix_dict = {'WHU': 'tif',
               'NewInria': 'png',
               'CrowdAI': 'jpg',
               'Mass': 'png',
               'NZ32': 'png',
               'CTC': 'png',
               'vaihingen': 'png',
               'potsdam': 'png',}

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]
k = 1

normalize_fn = Normalize(imagenet_mean, imagenet_std)

train_transforms = [
    # RandomResize(args.min_image_size, args.max_image_size),
    RandomHorizontalFlip(),
]

train_transform = transforms.Compose(
        [
            Normalize(imagenet_mean, imagenet_std),
            Transpose()
        ]
    )
test_transform = transforms.Compose([
    Normalize(imagenet_mean, imagenet_std),
])


class BuildingDataset(Dataset):
    def __init__(self, split, dataset_name ='WHU'):
        assert dataset_name in ['WHU', 'NewInria', 'CrowdAI', 'Mass','NZ32','CTC','vaihingen','potsdam','WHU256'], \
            'dataset_name must be in [WHU, NewInria, CrowdAI, Mass, NZ32, CTC]'
        self.suffix = suffix_dict[dataset_name]
        dataset_dict = {'WHU': r'C:\ZTB\Dataset\WHUBuilding',
                        'NewInria':r'C:\ZTB\Dataset\NewInria',
                        'Mass':r'E:\DataSet\Massachusetts',
                        'CrowdAI':r'C:\ZTB\Dataset\CrowdAI_split',
                        'NZ32':r'E:\DataSet\NZ32km2',
                        'CTC':r'C:\ZTB\Dataset\CTC',
                        'vaihingen':r'C:\ZTB\Dataset\Vaihingen_wsss',
                        'potsdam':r'C:\ZTB\Dataset\Potsdam_wsss',}
        self.path = dataset_dict[dataset_name]
        self.split = split
        self.names = read_image_names(self.path, self.split)

    def __getitem__(self, idx):
        image_path = os.path.join(self.path, self.split, 'image', f'{self.names[idx]}.{self.suffix}')
        label_path = os.path.join(self.path, self.split, 'label', f'{self.names[idx]}.{self.suffix}')
        edge_path = os.path.join(self.path, self.split, 'edge', f'{self.names[idx]}.{self.suffix}')
        label = Image.open(label_path)
        edge = Image.open(edge_path) if not self.split in ['test'] else None
        image = to_tensor(Image.open(image_path))
        tensor_label = to_tensor(label)  # .long()
        tensor_label[tensor_label > 0.5] = 1
        tensor_label[tensor_label < 1] = 0
        if edge is not None:
            tensor_edge = to_tensor(edge)
            tensor_edge[tensor_edge > 0.5] = 1
            tensor_edge[tensor_edge < 1] = 0
        else:
            tensor_edge = None

        if self.split == 'test':
            return image, tensor_label, self.names[idx]
        else:
            return image, tensor_label, tensor_edge

    def get_cls_label(self, tensor_label):
        cls_label = torch.zeros(2, dtype=torch.float32)
        if torch.sum(tensor_label) == 0:
            cls_label[0] = 1.0
        else:
            cls_label[1] = 1.0
        return cls_label

    def __len__(self):
        return len(self.names)

class WSBuildingDataset(BuildingDataset):
    # 还需要加入CAM路径
    def __init__(self, split, dataset_name='potsdam', inflated=False, counting=False, classification=False,
                 gen_cam=False, affinity=False, scribble=False, crop_size=256, aff_index=None):
        if classification:
            assert dataset_name in ['potsdam', 'vaihingen', 'WHU256'], 'For classification, dataset_name must be in [potsdam, vaihingen, WHU256]'
        super().__init__(split, dataset_name)
        self.inflated = inflated
        if self.inflated:
            self.names = read_image_names(self.path, split, folder='inflated_image')
        if classification and split=='train':  # For classification, remove images with (0,25) building coverage
            self.names = [name for name in self.names if not name.endswith('expel')]
        self.counting = counting
        self.classification = classification
        self.crop_size = crop_size
        self.gen_cam = gen_cam
        self.gen_aff = affinity
        self.path_index = aff_index
        self.buildings_count = pd.read_csv(
            os.path.join(self.path, self.split, 'BuildingCounts.csv')) if self.counting else None

        # 为训练集添加数据增强
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=90),

                # transforms.RandomCrop(crop_size),
                transforms.Resize((self.crop_size//4,self.crop_size//4)) if affinity else (lambda x: x),
                transforms.ToTensor()
            ])
        else:
            # 验证集和测试集不使用数据增强
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])



    def __getitem__(self, idx):
        if self.inflated:
            image_path = os.path.join(self.path, self.split, 'inflated_image', f'{self.names[idx]}.{self.suffix}')
            pure_name, building_count = self.names[idx].split('.')[0].split('_')
            self.building_count = int(building_count)

            label_path = os.path.join(self.path, self.split, 'label', f'{pure_name}.{self.suffix}')
            edge_path = os.path.join(self.path, self.split, 'edge', f'{pure_name}.{self.suffix}')
        else:
            image_path = os.path.join(self.path, self.split, 'image', f'{self.names[idx]}.{self.suffix}')
            label_path = os.path.join(self.path, self.split, 'label', f'{self.names[idx]}.{self.suffix}')
            edge_path = os.path.join(self.path, self.split, 'edge', f'{self.names[idx]}.{self.suffix}')
        if self.counting:
            image = to_tensor(Image.open(image_path))
            count = torch.tensor(
                self.buildings_count[self.buildings_count['image_name'] == f'{self.names[idx]}.{self.suffix}'][
                    'num_buildings'].iloc[0],
                dtype=torch.float32)
            return image, count, self.names[idx]
        elif self.classification:
            image = Image.open(image_path)
            label = Image.open(label_path)

            cls_label = torch.zeros(2, dtype=torch.float32)
            if self.inflated:
                if self.building_count == 0:
                    cls_label[0] = 1.0
                    cls_label[1] = 0.0
                elif self.building_count > 0:
                    cls_label[1] = 1.0
                    cls_label[0] = 0.0
            else:
                if self.names[idx].endswith('non'):
                    cls_label[0] = 1.0
                    cls_label[1] = 0.0
                elif self.names[idx].endswith('exist'):
                    cls_label[1] = 1.0
                    cls_label[0] = 0.0
            if self.split == 'train':
                return self.transform(image), cls_label
            else:
                tensor_label = to_tensor(label)
                tensor_label[tensor_label > 0.5] = 1
                tensor_label[tensor_label < 1] = 0
                return to_tensor(image), cls_label, tensor_label
        elif self.gen_cam:
            label = np.array([0,0])
            if self.names[idx].endswith('non'):
                label[0] = 1.0
                label[1] = 0.0
            else:
                label[1] = 1.0
                label[0] = 0.0
            return (np.array(Image.open(image_path))/255.0,
                    self.names[idx],
                    label, np.array(Image.open(label_path))/255.0)

        elif self.gen_aff:
            aff_path = os.path.join(self.path, self.split, 'affinity', f'{self.names[idx]}.{self.suffix}')
            image = Image.open(image_path)
            aff_label = Image.open(aff_path)

            tensor_image = to_tensor(image)
            np_aff_label = np.array(aff_label)

            extract_aff_lab_func = GetAffinityLabelFromIndices(self.path_index.src_indices,
                                                                    self.path_index.dst_indices)
            return tensor_image, extract_aff_lab_func(np_aff_label)
        else:
            image = to_tensor(Image.open(image_path))
            label = Image.open(label_path)
            edge = Image.open(edge_path) if not self.split in ['test'] else None
            tensor_label = to_tensor(label)  # .long()
            tensor_label[tensor_label > 0.5] = 1
            tensor_label[tensor_label < 1] = 0
            if edge is not None:
                tensor_edge = to_tensor(edge)
                tensor_edge[tensor_edge > 0.5] = 1
                tensor_edge[tensor_edge < 1] = 0
            else:
                tensor_edge = None

            if self.split == 'test':
                return image, tensor_label, self.names[idx]
            else:
                return image, tensor_label, tensor_edge, self.names[idx]