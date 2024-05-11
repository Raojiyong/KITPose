from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import random
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import numpy as np

import _init_paths
from config import cfg
from config import update_config
from core.loss import HeatmapLoss
from core.function_cutmix_part import ensemble_validate
from utils.utils import create_logger
from collections import OrderedDict
import json
import math
from scipy.io import savemat, loadmat

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


def model_loader(cfg, path_to_pth):
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )

    model_state_file = os.path.join(
        path_to_pth, 'model_best.pth'
    )
    model.load_state_dict(torch.load(model_state_file))
    return model


def test_model(cfg, model):
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    # define loss function (criterion) and optimizer
    critetion_kpt = HeatmapLoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

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
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU * len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )
    result = ensemble_validate(cfg, valid_loader, valid_dataset, model, critetion_kpt, final_output_dir)
    return result


def evaluate(cfg, preds, output_dir):
    SC_BIAS = 1
    threshold = 0.1

    gt_file = os.path.join(cfg.DATASET.ROOT,
                           'pose_estimation/annotation/ak_P1',
                           '{}.json'  # gt_{}.json'
                           .format(cfg.DATASET.TEST_SET)
                           )

    with open(gt_file) as f:
        gt_dict = json.load(f)

    ### Changes below ###

    # dataset_joints = gt_dict['dataset_joints']
    # jnt_visible = [v for k, v in gt_dict['joints_vis'].items()]
    # pos_gt_src = [v for k, v in gt_dict['joints'].items()]
    # scale = [v for k, v in gt_dict['scale'].items()]

    dataset_joints = [
        [
            "Head_Mid_Top"
        ],
        [
            "Eye_Left"
        ],
        [
            "Eye_Right"
        ],
        [
            "Mouth_Front_Top"
        ],
        [
            "Mouth_Back_Left"
        ],
        [
            "Mouth_Back_Right"
        ],
        [
            "Mouth_Front_Bottom"
        ],
        [
            "Shoulder_Left"
        ],
        [
            "Shoulder_Right"
        ],
        [
            "Elbow_Left"
        ],
        [
            "Elbow_Right"
        ],
        [
            "Wrist_Left"
        ],
        [
            "Wrist_Right"
        ],
        [
            "Torso_Mid_Back"
        ],
        [
            "Hip_Left"
        ],
        [
            "Hip_Right"
        ],
        [
            "Knee_Left"
        ],
        [
            "Knee_Right"
        ],
        [
            "Ankle_Left"
        ],
        [
            "Ankle_Right"
        ],
        [
            "Tail_Top_Back"
        ],
        [
            "Tail_Mid_Back"
        ],
        [
            "Tail_End_Back"
        ]
    ]

    jnt_visible = [x['joints_vis'] for x in gt_dict]
    pos_gt_src = [x['joints'] for x in gt_dict]
    scale = [x['scale'] for x in gt_dict]

    scale = np.array(scale)
    scale = scale * 200 * math.sqrt(2)

    jnt_visible = np.transpose(jnt_visible, [1, 0])
    pos_pred_src = np.transpose(preds, [1, 2, 0])
    pos_gt_src = np.transpose(pos_gt_src, [1, 2, 0])
    dataset_joints = np.array(dataset_joints)
    head = np.where(dataset_joints == 'Head_Mid_Top')[0][0]
    lsho = np.where(dataset_joints == 'Shoulder_Left')[0][0]
    lelb = np.where(dataset_joints == 'Elbow_Left')[0][0]
    lwri = np.where(dataset_joints == 'Wrist_Left')[0][0]
    lhip = np.where(dataset_joints == 'Hip_Left')[0][0]
    lkne = np.where(dataset_joints == 'Knee_Left')[0][0]
    lank = np.where(dataset_joints == 'Ankle_Left')[0][0]

    rsho = np.where(dataset_joints == 'Shoulder_Right')[0][0]
    relb = np.where(dataset_joints == 'Elbow_Right')[0][0]
    rwri = np.where(dataset_joints == 'Wrist_Right')[0][0]
    rhip = np.where(dataset_joints == 'Hip_Right')[0][0]
    rkne = np.where(dataset_joints == 'Knee_Right')[0][0]
    rank = np.where(dataset_joints == 'Ankle_Right')[0][0]

    tmouth = np.where(dataset_joints == 'Mouth_Front_Top')[0][0]
    lmouth = np.where(dataset_joints == 'Mouth_Back_Left')[0][0]
    rmouth = np.where(dataset_joints == 'Mouth_Back_Right')[0][0]
    bmouth = np.where(dataset_joints == 'Mouth_Front_Bottom')[0][0]
    ttail = np.where(dataset_joints == 'Tail_Top_Back')[0][0]
    mtail = np.where(dataset_joints == 'Tail_Mid_Back')[0][0]
    btail = np.where(dataset_joints == 'Tail_End_Back')[0][0]

    uv_error = pos_pred_src - pos_gt_src

    uv_err = np.linalg.norm(uv_error, axis=1)

    #         headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    #         headsizes = np.linalg.norm(headsizes, axis=0)
    scale *= SC_BIAS
    headsizes = scale

    scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
    scaled_uv_err = np.divide(uv_err, scale)
    scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
    jnt_count = np.sum(jnt_visible, axis=1)

    less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                      jnt_visible)
    PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)
    # save
    rng = np.arange(0, 0.5 + 0.01, 0.01)
    pckAll = np.zeros((len(rng), 23))

    for r in range(len(rng)):
        threshold = rng[r]
        less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                          jnt_visible)
        pckAll[r, :] = np.divide(100. * np.sum(less_than_threshold, axis=1),
                                 jnt_count)

    #         PCKh = np.ma.array(PCKh, mask=False)
    #         PCKh.mask[21:22] = True

    #         jnt_count = np.ma.array(jnt_count, mask=False)
    #         jnt_count.mask[21:22] = True

    jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)
    name_value = [
        ('Head', PCKh[head]),
        ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
        ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
        ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
        ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
        ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
        ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
        ('Mouth', 0.25 * (PCKh[tmouth] + PCKh[lmouth] + PCKh[rmouth] + PCKh[bmouth])),
        ('Tail', (PCKh[ttail] + PCKh[mtail] + PCKh[btail]) / 3),
        ('Mean', np.sum(PCKh * jnt_ratio))
        #             ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
    ]
    name_value = OrderedDict(name_value)
    return name_value, name_value['Mean']


def mean_ensemble():
    model_paths = [
        # '/dataQ/jiyong/kitpose/output/ak/ak/kitpose_part/2Encoder_w48_256x256_adaW_cutmix_p4_all',
        # '/dataQ/jiyong/kitpose/output/ak/ak/kitpose_part/2Encoder_w48_256x256_adaW_cutmix_p4_P1S1',
        # '/dataQ/jiyong/kitpose/output/ak/ak/kitpose_part/2Encoder_w48_256x256_adaW_cutmix_p4_P1S2',
        # '/dataQ/jiyong/kitpose/output/ak/ak/kitpose_part/4Encoder_w48_256x256_adaW_cutmix_p4_P1S3',
        '/dataQ/jiyong/kitpose/output/ak/ak/kitpose_part/2Encoder_w48_256x256_adaW_cutmix_p4_P1S4',
        '/dataQ/jiyong/kitpose/output/ak/ak/kitpose_part/2Encoder_w48_256x256_adaW_cutmix_p4_P1S5',
    ]

    models = [model_loader(cfg, path) for path in model_paths]
    outcomes = [test_model(cfg, m) for m in models]

    sum_keypoints = np.zeros((len(outcomes[0]), 23, 2))
    for pred in outcomes:
        sum_keypoints += pred[:, :, :2]
    sum_keypoints = sum_keypoints / len(outcomes)
    pred_file = os.path.join(final_output_dir, 'pred.mat')
    savemat(pred_file, mdict={'pred': sum_keypoints})

    name_values, per_indicator = evaluate(cfg, sum_keypoints, final_output_dir)
    model_name = cfg.MODEL.NAME
    if isinstance(name_values, list):
        for name_value in name_values:
            _print_name_value(name_value, model_name)
    else:
        _print_name_value(name_values, model_name)


def jsonparser(path, num_keypoints):
    # Read the JSON
    import numpy as np
    with open(path, 'r') as jf:
        data = jf.read()
    import json
    json = json.loads(data)

    # Figure out the keys in the annotation
    keys = []
    for key, values in json.items():
        keys.append(key)
    ann_key = keys.index('annotations')
    im_key = keys.index('images')

    # create the placeholders
    imgname = []  # shape 1xn
    bndbox = []  # shape [1x4]x n images
    part = []  # shape [17x2]x n images

    for annotations in json[keys[ann_key]]:
        box = annotations['bbox']
        keypts = np.array(annotations['keypoints'])
        # print(keypts)
        # keypts = np.delete(keypts, np.arange(2, keypts.size, 3))
        bndbox.append(box)
        part.append(keypts)
    # Bounding boxes
    bndbox = np.array(bndbox)
    # bndbox = np.reshape(bndbox,(-1, 1, 4))

    # keypoint locations
    # part = np.array(part)
    # part = np.reshape(part,(-1,15,2))

    for image in json[keys[im_key]]:
        asciiname = image['file_name']
        asciiname = asciiname.split('.')[0]
        # asciiname = ([ord(c) for c in asciiname])
        imgname.append(asciiname)

    # imgname = np.array(imgname)

    return imgname, bndbox, part


def bagging_ensemble():
    model_paths = [
        '/dataQ/jiyong/kitpose/output/ak/ak/kitpose_part/2Encoder_w48_256x256_adaW_cutmix_p4_P1S1',
        '/dataQ/jiyong/kitpose/output/ak/ak/kitpose_part/2Encoder_w48_256x256_adaW_cutmix_p4_P1S2',
        '/dataQ/jiyong/kitpose/output/ak/ak/kitpose_part/2Encoder_w48_256x256_adaW_cutmix_p4_P1S3',
        '/dataQ/jiyong/kitpose/output/ak/ak/kitpose_part/2Encoder_w48_256x256_adaW_cutmix_p4_P1S4',
        '/dataQ/jiyong/kitpose/output/ak/ak/kitpose_part/2Encoder_w48_256x256_adaW_cutmix_p4_P1S5',
    ]
    gt_file = os.path.join(cfg.DATASET.ROOT,
                           'pose_estimation/annotation/ak_P1',
                           '{}.json'  # gt_{}.json'
                           .format(cfg.DATASET.TEST_SET)
                           )

    with open(gt_file) as f:
        gt_dict = json.load(f)
    gt_imIds, gt_bbox, gt_kpt = jsonparser('data/ak/pose_estimation/annotation_coco/ak_P1/test.json', 23)
    jnt_visible = [x['joints_vis'] for x in gt_dict]
    pos_gt_src = [x['joints'] for x in gt_dict]
    jnt_visible = np.expand_dims(np.transpose(jnt_visible, [0, 1]), axis=-1)
    pos_gt_src = np.transpose(pos_gt_src, [0, 1, 2])

    models = [model_loader(cfg, path) for path in model_paths]
    outcomes = [test_model(cfg, m) for m in models]

    train_x, train_y = [], []
    pos_gt_src = np.concatenate((pos_gt_src, jnt_visible), axis=2).reshape(-1, 69)
    for pred in outcomes:
        pred[:, :, 2:] = jnt_visible
        pred = pred.reshape(-1, 69)
        train_x.append(pred)
        train_y.append(pos_gt_src)

    from sklearn.ensemble import BaggingRegressor
    from sklearn import tree

    model = BaggingRegressor(tree.DecisionTreeRegressor(random_state=1))
    model.fit(train_x, train_y)

    test_model_path = 'output/ak/ak/kitpose_part/2Encoder_w48_256x256_adaW_cutmix_p4_P1S4'
    model_test = model_loader(cfg, test_model_path)
    test_outcomes = test_model(cfg, model_test)

    for idx in range(len(test_outcomes)):
        test_x = test_outcomes[idx]

    # name_values, per_indicator = evaluate(cfg, sum_keypoints, final_output_dir)
    # model_name = cfg.MODEL.NAME
    # if isinstance(name_values, list):
    #     for name_value in name_values:
    #         _print_name_value(name_value, model_name)
    # else:
    #     _print_name_value(name_values, model_name)


def _print_name_value(name_value, full_arch_name):
    names = name_value.keys()
    values = name_value.values()
    num_values = len(name_value)
    logger.info(
        '| Arch ' +
        ' '.join(['| {}'.format(name) for name in names]) +
        ' |'
    )
    logger.info('|---' * (num_values + 1) + '|')

    if len(full_arch_name) > 15:
        full_arch_name = full_arch_name[:8] + '...'
    logger.info(
        '| ' + full_arch_name + ' ' +
        ' '.join(['| {:.3f}'.format(value) for value in values]) +
        ' |'
    )


if __name__ == '__main__':
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
    final_output_dir = 'output/ak/ak/kitpose_part/mean_ensemble_test'
    mean_ensemble()
    # bagging_ensemble()
