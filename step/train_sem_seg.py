import torch
from torch.backends import cudnn

cudnn.enabled = True
from torch.utils.data import DataLoader
import voc12.dataloader
from misc import pyutils, torchutils, indexing
import importlib
from PIL import ImageFile
import sys
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

ImageFile.LOAD_TRUNCATED_IMAGES = True


def evaluate(model, loader):
    model.eval()

    meter = Calculator_For_mIoU()

    with torch.no_grad():
        length = len(loader)
        for step, (images, labels) in enumerate(loader):
            images = images.cuda()
            labels = labels.cuda()

            logits = model(images)
            predictions = torch.argmax(logits, dim=1)

            for batch_index in range(images.size()[0]):
                pred_mask = get_numpy_from_tensor(predictions[batch_index])
                gt_mask = get_numpy_from_tensor(labels[batch_index])

                h, w = pred_mask.shape
                gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                meter.add(pred_mask, gt_mask)

            sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
            sys.stdout.flush()

    print(' ')
    model.train()

    return meter.get(clear=True)


def run(args):
    log_dir = args.log_dir

    model = DeepLabv3_Plus(args.backbone, num_classes=1, mode=args.mode,
                           use_group_norm=args.use_gn)
    param_groups = model.get_parameter_groups(None)
    params = [
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2 * args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20 * args.lr, 'weight_decay': 0},
    ]

    model = model.cuda()
    model.train()