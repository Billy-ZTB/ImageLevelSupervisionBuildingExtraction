import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from core.networks import DeepLabv3_Plus
from voc12.dataloader import *
from torch.utils.data import DataLoader
from chainercv.evaluations import calc_semantic_segmentation_confusion
from PIL import Image

parser = argparse.ArgumentParser(description='Test weakly supervised semantic segmentation model')
parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
parser.add_argument('--backbone', type=str, default='resnet50', help='Backbone architecture (default: resnet50)')
parser.add_argument('--voc12_root', type=str, required=True, help='Path to the dataset root')
parser.add_argument('--test_list', type=str, required=True, help='Path to the test image list')
parser.add_argument('--output_dir', type=str, required=True, help='Directory to save the output predictions')
args = parser.parse_args()

if __name__ == '__main__':
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    model = DeepLabv3_Plus(model_name=args.backbone,num_classes=1, use_group_norm=True).to(DEVICE)
    weights = torch.load(args.model_path, map_location=DEVICE)
    model.load_state_dict(weights)
    model.eval()

    test_dataset = VOC12SegmentationDataset(img_name_list_path=args.test_list, voc12_root=args.voc12_root, label_dir='',
                                            crop_size=256, split='test', crop_method='none')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    pred_list, label_list = [], []
    pred_dir = args.output_dir+'/predictions/'
    os.makedirs(pred_dir, exist_ok=True)
    for idx, data in enumerate(test_loader):
        img = data['img'].to(DEVICE)
        img_name = data['name'][0]
        label = data['label']

        with torch.no_grad():
            output = model(img)
            output = torch.sigmoid(output).squeeze()
            output = torch.where(output > 0.5, 1, 0)
        output = output.cpu().long().squeeze().numpy()
        label = label.squeeze().numpy().astype(np.int64)
        pred_list.append(output)
        label_list.append(label)
        #print('pred shape:', output.shape,'label shape:', label.shape)
        save_output = Image.fromarray(np.uint8(output*255))
        save_output.save(os.path.join(args.output_dir, img_name+'.png'))
        print(f'Saved prediction for {img_name} to {args.output_dir}')

    confusion = calc_semantic_segmentation_confusion(pred_list, label_list)[:2, :2]
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator

    iou = gtjresj / denominator
    precision = gtjresj / (fp * denominator + gtjresj)
    recall = gtjresj / (fn * denominator + gtjresj)
    F_score = 2 * (precision * recall) / (precision + recall)

    print({'precision': precision, 'recall': recall, 'F_score': F_score,'IoU': iou, 'mIoU': np.nanmean(iou)})
    with open(f'{args.output_dir}/metrics.txt', 'w') as f:
        f.write(f'Precision: {precision}\n')
        f.write(f'Recall: {recall}\n')
        f.write(f'F_score: {F_score}\n')
        f.write(f'IoU: {iou}\n')
        f.write(f'mIoU: {np.nanmean(iou)}\n')
