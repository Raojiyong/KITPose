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
from torch import distributed as dist
import torchvision.transforms as transforms

import _init_paths
from config import cfg_moe as cfg
from config import update_config_moe as update_config
from core.loss import HeatmapLoss, AdaptiveLoss, BPLoss, GFL_5x5
from core.function_cutmix_part_moe import train_cutmix
from core.function_cutmix_part_moe import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary
from utils.collate import collate
from utils.distributed_sampler import DistributedSampler
from functools import partial

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


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def build_datasets(cfg, root_cfg, is_train=True, idx=0):
    from dataset.ConcatDataset import ConcatDataset
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([build_datasets(c, root_cfg, is_train=is_train, idx=i) for i, c in enumerate(cfg)])
    else:
        dataset = eval('dataset.' + cfg['name'])(
            root_cfg, cfg['root'], cfg['train_set'] if is_train else cfg['test_set'], is_train,
            transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            data_cfg=cfg
        )
    return dataset


def _get_dataloader():
    from torch.utils.data import DataLoader
    PoolDataLoader = DataLoader
    return DataLoader, PoolDataLoader


def worker_init_fn(worker_id, num_workers, rank, seed):
    """Init the random seed for various workers."""
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloader(dataset, samples_per_gpu, workers_per_gpu, num_gpus=1, dist=False,
                     shuffle=True, seed=None, drop_last=True, pin_memory=True, **kwargs):
    rank, world_size = get_dist_info()
    if dist:
        sampler = DistributedSampler(
            dataset, world_size, rank, shuffle=shuffle, seed=seed, batch_size=samples_per_gpu)
        shuffle = False
        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
    else:
        sampler = None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu

    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank, seed=seed
    ) if seed is not None else None

    _, Dataloader = _get_dataloader()

    dataloader = Dataloader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=pin_memory,
        shuffle=shuffle,
        worker_init_fn=init_fn,
        drop_last=drop_last,
        **kwargs)

    return dataloader


def main():
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=True
    )

    # copy model file
    this_dir = os.path.dirname(__file__)
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )

    logger.info(get_model_summary(model, dump_input, num_experts=cfg.MODEL.NUM_EXPERTS))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    criterion_kpt = HeatmapLoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    criterion_part = GFL_5x5().cuda()

    # Data loading code

    train_datasets = [build_datasets(cfg.DATASET.MIX_DATA.CONFIG, cfg)]
    # valid_datasets = [build_datasets(cfg.DATASET.MIX_DATA.CONFIG, cfg, is_train=False)]
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    train_loader_cfg = {
        **dict(
            seed=cfg.SEED, drop_last=False, dist=True, num_gpus=len(cfg.GPUS),
            samples_per_gpu=cfg.TRAIN.BATCH_SIZE_PER_GPU, workers_per_gpu=cfg.WORKERS,
            shuffle=cfg.TRAIN.SHUFFLE, pin_memory=cfg.PIN_MEMORY
        )
    }
    train_loaders = [build_dataloader(ds, **train_loader_cfg) for ds in train_datasets]

    # valid_loader_cfg = {
    #     **dict(
    #         seed=cfg.SEED, drop_last=False, dist=True, num_gpus=len(cfg.GPUS),
    #         samples_per_gpu=cfg.TEST.BATCH_SIZE_PER_GPU, workers_per_gpu=cfg.WORKERS,
    #         shuffle=False, pin_memory=cfg.PIN_MEMORY
    #     )
    # }
    # valid_loaders = [build_dataloader(ds, **valid_loader_cfg) for ds in valid_datasets]
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    # valid_dataloader = build_dataloader(valid_dataset, **valid_loader_cfg)

    best_perf = 0.0
    best_model = False
    last_epoch = -1

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

    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):

        logger.info("=> the first lr is {:.6f}, the second lr is {:.6f}".format(
            lr_scheduler.get_last_lr()[0],
            lr_scheduler.get_last_lr()[1]))

        # train for one epoch
        wandb_train_log = train_cutmix(cfg, train_loaders, model, criterion_kpt, criterion_part, optimizer, epoch,
                                       final_output_dir, wandb)

        # evaluate on validation set
        perf_indicator, wandb_val_log = validate(
            cfg, valid_loader, valid_dataset, model, criterion_kpt,
            final_output_dir, wandb, log_type=cfg.LOG_TYPE
        )
        lr_scheduler.step()

        # update wandb
        wandb_log.update(wandb_train_log)
        wandb_log.update(wandb_val_log)
        wandb_log['lr'] = lr_scheduler.get_last_lr()[0]
        wandb_log['epoch'] = epoch
        wandb_log['best_map'] = best_perf
        wandb.log(
            data=wandb_log,
            step=epoch,
        )

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)


if __name__ == '__main__':
    args = parse_args()
    update_config(cfg, args)
    seed_everything(cfg.SEED)

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

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    main()
