import torch
from torch.backends import cudnn

from voc12.dataloader import VOC12SegmentationDataset

cudnn.enabled = True
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
from torch.cuda.amp import autocast as autocast


ImageFile.LOAD_TRUNCATED_IMAGES = True


def evaluate(model, loader):
    model.eval()

    meter = Calculator_For_mIoU()

    with torch.no_grad():
        length = len(loader)
        for step, data in enumerate(loader):
            images = data['img'].cuda()
            labels = data['label'].cuda()

            logits = model(images)
            predictions = (torch.sigmoid(logits)>0.5).float()

            for batch_index in range(images.size()[0]):
                pred_mask = get_numpy_from_tensor(predictions[batch_index].squeeze())
                gt_mask = get_numpy_from_tensor(labels[batch_index])

                h, w = pred_mask.shape
                gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                meter.add(pred_mask, gt_mask)

            sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
            sys.stdout.flush()
    print(' ')
    model.train()
    return meter.get(detail=True, clear=True)


def run(args):
    result_dir = './result/semantic_segmentation/'  # create_directory(f'./experiments/{args.tag}/logs/')
    log_path = result_dir + f'{args.tag}/logs.txt'
    log_func = lambda string='': log_print(string, log_path)
    model_output_dir = create_directory(result_dir + f'/{args.tag}/models/')
    model_path = model_output_dir + f'{args.tag}.pth'
    tensorboard_dir = create_directory(result_dir + f'{args.tag}/tensorboard/')
    data_dir = create_directory(result_dir + f'{args.tag}/data/')
    data_path = data_dir + f'{args.tag}.json'
    log_func('[i] {}'.format(args.tag))
    log_func()

    set_seed(100)
    model = DeepLabv3_Plus(args.pss_backbone, num_classes=1, use_group_norm=True)

    param_groups = model.get_parameter_groups(None)
    params = [
        {'params': param_groups[0], 'lr': args.pss_lr, 'weight_decay': args.pss_wd},
        {'params': param_groups[1], 'lr': 2 * args.pss_lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10 * args.pss_lr, 'weight_decay': args.pss_wd},
        {'params': param_groups[3], 'lr': 20 * args.pss_lr, 'weight_decay': 0},
    ]
    model = model.cuda()
    model.train()

    train_dataset = VOC12SegmentationDataset(args.train_list, args.pss_label_dir, args.pss_crop_size, args.voc12_root, )
    train_loader = DataLoader(train_dataset, batch_size=args.pss_batch_size, shuffle=True, num_workers=8)
    valid_dataset = VOC12SegmentationDataset(args.valid_list, args.pss_label_dir, args.pss_crop_size, args.voc12_root,
                                             crop_method='none', split='valid')
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.pss_num_epochs * val_iteration

    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn_for_backup = lambda: save_model(model, model_path.replace('.pth', f'_backup.pth'),
                                                  parallel=the_number_of_gpu > 1)

    class_loss_fn = nn.BCEWithLogitsLoss().cuda()

    optimizer = PolyOptimizer(params, lr=args.pss_lr, momentum=0.9, weight_decay=args.pss_wd, max_step=max_iteration,
                              nesterov=True)

    scaler = torch.cuda.amp.GradScaler()

    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train': [],
        'validation': [],
    }

    train_timer = Timer()
    eval_timer = Timer()

    train_meter = Average_Meter(['loss'])

    best_valid_mIoU = -1
    best_valid_mIoU_fg = -1

    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)

    torch.autograd.set_detect_anomaly(True)

    for iteration in range(max_iteration):
        batch = train_iterator.get()
        images = batch['img']
        labels = batch['label']
        images, labels = images.cuda(), labels.cuda()

        #################################################################################################
        # Inference
        #################################################################################################
        optimizer.zero_grad()

        # 前向过程(model + loss)开启 autocast
        with autocast():
            logits = model(images)

            loss = class_loss_fn(logits, labels.unsqueeze(1).float())

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        train_meter.add({
            'loss': loss.item(),
        })

        #################################################################################################
        # For Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            loss = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))

            data = {
                'iteration': iteration + 1,
                'learning_rate': learning_rate,
                'loss': loss,
                'time': train_timer.tok(clear=True),
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

        #################################################################################################
        # Evaluation
        #################################################################################################
        if (iteration + 1) % val_iteration == 0:
            mIoU, mIoU_fg, IoU_dic, FP, FN = evaluate(model,valid_loader)

            if best_valid_mIoU == -1 or best_valid_mIoU < mIoU:
                best_valid_mIoU = mIoU

            if best_valid_mIoU_fg == -1 or best_valid_mIoU_fg < mIoU_fg:
                best_valid_mIoU_fg = mIoU_fg

                save_model_fn()
                log_func('[i] save model')

            data = {
                'iteration': iteration + 1,
                'mIoU': mIoU,
                'best_valid_mIoU': best_valid_mIoU,
                'mIoU_fg': mIoU_fg,
                'best_valid_mIoU_fg': best_valid_mIoU_fg,
                'time': eval_timer.tok(clear=True),
            }
            data_dic['validation'].append(data)
            write_json(data_path, data_dic)

            log_func('[i] \
                    iteration={iteration:,}, \
                    mIoU={mIoU:.2f}%, \
                    best_valid_mIoU={best_valid_mIoU:.2f}%, \
                    IoU_fg={mIoU_fg:.2f}%, \
                    best_valid_mIoU_fg={best_valid_mIoU_fg:.2f}%, \
                    time={time:.0f}sec'.format(**data)
                     )

            writer.add_scalar('Evaluation/mIoU', mIoU, iteration)
            writer.add_scalar('Evaluation/best_valid_mIoU', best_valid_mIoU, iteration)
            writer.add_scalar('Evaluation/mIoU_fg', mIoU_fg, iteration)
            writer.add_scalar('Evaluation/best_valid_mIoU_fg', best_valid_mIoU_fg, iteration)
            writer.add_scalar('Evaluation/FP', FP, iteration)
            writer.add_scalar('Evaluation/FN', FN, iteration)
    write_json(data_path, data_dic)
    writer.close()

    print(args.tag)