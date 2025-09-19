import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
from voc12.dataloader import *
from chainercv.evaluations import calc_semantic_segmentation_confusion
from model_loss_semseg_gatedcrf import ModelLossSemsegGatedCRF
from DenseEnergyLoss import DenseEnergyLoss
from hrnet import HRNet
from torch.cuda.amp import autocast as autocast

parser = argparse.ArgumentParser()


def evaluate(model, loader):
    model.eval()

    pred_list, label_list = [], []

    with torch.no_grad():
        length = len(loader)
        for step, data in enumerate(loader):
            image = data['img']
            label = data['label'].numpy().astype(np.int64)
            name = data['name'][0]

            logits = model(image)
            prediction = torch.sigmoid(logits).squeeze()
            prediction = torch.where(prediction > 0.5, 1, 0).cpu().detach().numpy().astype(np.int64)

            pred_list.append(prediction)
            label_list.append(label)

            pred2save = Image.fromarray(np.uint8(prediction))
            pred2save.save(os.path.join(args.sem_seg_out_dir, name + '.png'))

    confusion = calc_semantic_segmentation_confusion(pred_list, label_list)[:2, :2]
    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    fp = 1. - gtj / denominator
    fn = 1. - resj / denominator
    iou = gtjresj / denominator
    print("total images", length)
    precision = gtjresj / (fp * denominator + gtjresj)
    recall = gtjresj / (fn * denominator + gtjresj)
    F_score = 2 * (precision * recall) / (precision + recall)
    print({'precision': precision, 'recall': recall, 'F_score': F_score})
    print({'iou': iou, 'miou': np.nanmean(iou)})
    return {'precision': precision, 'recall': recall, 'F_score': F_score, 'iou': iou, 'miou': np.nanmean(iou)}


def run(args):
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)

    if not os.path.exists(args.sem_seg_out_dir):
        os.makedirs(args.sem_seg_out_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepLabv3_Plus(model_name=args.backbone, num_classes=1, use_group_norm=True).to(device)
    param_groups = model.get_parameter_groups(None)
    params = [
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0},
    ]
    model.train()

    data_dic = {
        'train': [],
        'validation': [],
    }
    train_dataset = VOC12SegmentationDataset(img_name_list_path=args.train_list, voc12_root=args.voc12_root,
                                             label_dir=args.sem_seg_out_dir,
                                            crop_size=256, split='train', crop_method='none')
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)
    valid_dataset = VOC12SegmentationDataset(img_name_list_path=args.test_list, voc12_root=args.voc12_root,
                                             label_dir='',
                                            crop_size=256, split='valid', crop_method='none')
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.pss_epochs * val_iteration

    save_model_fn = lambda: save_model(model, args.pss_pth, parallel=False)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = PolyOptimizer(params, lr=args.pss_lr, momentum=0.9, weight_decay=args.wd, max_step=args.pss_epochs,
                              nesterov=True)
    train_meter = Average_Meter(['loss'])
    train_iterator = Iterator(train_loader)
    scaler = torch.cuda.amp.GradScaler()
    for i in range(max_iteration):
        print("PSS Epoch %d/%d" % (i + 1, args.pss_epochs))
        data = train_iterator.get()
        image = data['img'].to(device)
        label = data['label'].to(device).float()
        name = data['name'][0]

        with autocast():
            logits = model(image)
            loss = loss_fn(logits, label)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_meter.add({
            'loss': loss.item(),
        })

        if (i + 1) % log_iteration == 0:
            loss = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))

            data = {
                'iteration': i + 1,
                'learning_rate': learning_rate,
                'loss': loss,
            }
            data_dic['train'].append(data)
            write_json(data_path, data_dic)

            log_func('[i] \
                iteration={iteration:,}, \
                learning_rate={learning_rate:.4f}, \
                loss={loss:.4f}, \
                time={time:.0f}sec'.format(**data)
                     )

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)
