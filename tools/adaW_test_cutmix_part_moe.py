# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
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

import _init_paths
from config import cfg_moe as cfg
from config import update_config_moe as update_config
from core.loss import HeatmapLoss, BPLoss, GFL
from core.function_cutmix_part_moe import validate
from utils.utils import create_logger
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
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        load_file = torch.load(cfg.TEST.MODEL_FILE)
        model.load_state_dict(load_file, strict=False)
    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    # define loss function (criterion) and optimizer
    critetion_kpt = HeatmapLoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    criterion_part = BPLoss().cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    valid_dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_loader_cfg = {
        **dict(
            seed=cfg.SEED, drop_last=False, dist=True, num_gpus=len(cfg.GPUS),
            samples_per_gpu=cfg.TEST.BATCH_SIZE_PER_GPU, workers_per_gpu=cfg.WORKERS,
            shuffle=False, pin_memory=cfg.PIN_MEMORY
        )
    }
    # valid_loaders = [build_dataloader(ds, **valid_loader_cfg) for ds in valid_datasets]
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    # valid_dataloader = build_dataloader(valid_dataset, **valid_loader_cfg)

    # evaluate on validation set
    validate(cfg, valid_loader, valid_dataset, model, critetion_kpt,
             final_output_dir, tb_log_dir, log_type=cfg.LOG_TYPE)


if __name__ == '__main__':
    main()
