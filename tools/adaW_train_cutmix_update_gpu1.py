# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = "1,0,2"
import pprint
import shutil
import wandb
import random
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import HeatmapLoss
from core.function_cutmix_update import train_cutmix
from core.function_cutmix_update import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # wandb initialize
    config_dict = dict(
        yaml=args.cfg,
    )
    exp_name = args.cfg.split('/')[-1].split('.')[0]
    project_name = args.cfg.split('/')[1]
    wandb.init(
        project=project_name,
        name=exp_name,
        config=config_dict,
        entity="teamr",
    )
    wandb_log = dict()

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    seed = 22
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    # writer_dict['writer'].add_graph(model, (dump_input,))

    logger.info(get_model_summary(model, dump_input))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion = HeatmapLoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    train_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )

    best_perf = 0.0
    best_model = False
    last_epoch = -1

    # joints_weight = eval('dataset.' + cfg.DATASET.DATASET + '.ReturnWeight')(cfg)
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']

        writer_dict['train_global_steps'] = checkpoint['train_global_steps']
        writer_dict['valid_global_steps'] = checkpoint['valid_global_steps']

        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    ) if cfg.TRAIN.LR_SCHEDULER is 'MultiStepLR' else torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, cfg.TRAIN.END_EPOCH, eta_min=cfg.TRAIN.LR_END, last_epoch=last_epoch
    )

    model.cuda()

    # joints_weight_epoch = [i for i in range(100, cfg.TRAIN.END_EPOCH, 15)]
    # joints_weight.requires_grad = False
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        # joints_weight.requires_grad = False
        # for param in model.parameters():
        #     param.requires_grad = True
        # if epoch in joints_weight_epoch:
        #     joints_weight.requires_grad = True
        #     for param in model.parameters():
        #         param.requires_grad = False

        logger.info("=> the first lr is {:.6f}, the second lr is {:.6f}".format(
            lr_scheduler.get_last_lr()[0],
            lr_scheduler.get_last_lr()[1]))

        # train for one epoch
        wandb_train_log = train_cutmix(cfg, train_loader, model, criterion, optimizer, epoch,
                                       final_output_dir, tb_log_dir, wandb, writer_dict=writer_dict)
        # train(cfg, train_loader, model, criterion, optimizer, epoch, joints_weight,
        #       final_output_dir, tb_log_dir, writer_dict=writer_dict)

        # evaluate on validation set
        perf_indicator, wandb_val_log = validate(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, wandb, writer_dict, log_type=cfg.LOG_TYPE
        )
        # perf_indicator, _ = validate(
        #     cfg, valid_loader, valid_dataset, model, criterion,
        #     final_output_dir, joints_weight, tb_log_dir, writer_dict=writer_dict
        # )

        # update wandb
        wandb_log.update(wandb_train_log)
        wandb_log.update(wandb_val_log)
        wandb_log['lr'] = lr_scheduler.get_last_lr()[0]
        wandb_log['epoch'] = epoch
        wandb_log['best_map'