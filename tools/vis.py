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

# from visualizer import get_local
#
# get_local.activate()
# os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,0"
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as T

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from core.inference import get_final_preds
# from utils import transforms, vis
from utils.transforms import flip_back
from utils.utils import create_logger
from plot_attn import *
import cv2

import dataset
import models
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import matplotlib


# matplotlib.use('TkAgg')


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


def main():
    args = parse_args()
    update_config(cfg, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'visualization')

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    device = torch.device('cuda:0')
    model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        # for normal test
        load_file = torch.load(cfg.TEST.MODEL_FILE)
        model.load_state_dict(load_file, strict=False)
    else:
        logger.info('=> loading no model from ')
    model.to(device)

    # Data loading code
    normalize = T.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    dataset = eval('dataset.' + cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        T.Compose([
            T.ToTensor(),
            normalize,
        ])
    )
    global_atten_maps = []
    features = []
    kpt_tokens = []
    pos_emb = []
    feature_hooks = [
        model.pre_feature.final_layer.register_forward_hook(
            lambda self, input, output1: features.append(output1)
        )
    ]
    # tokens_hooks = [
    #     model.channelformer.transformer.layers[i].register_forward_hook(
    #         lambda self, input, output: kpt_tokens.append(output[0])
    #     )
    #     for i in range(len(model.channelformer.transformer.layers))
    # ]
    # atten_maps_hooks = [
    #     model.channelformer.transformer.layers[i].attn.register_forward_hook(
    #         lambda self, input, output: global_atten_maps.append(output[1])
    #     )
    #     for i in range(len(model.channelformer.transformer.layers))
    # ]

    with torch.no_grad():
        model.eval()
        i = 0
        idx = dataset.db[i]['image'].split('/')[-1].split('.')[0]
        img = dataset[i][0]  # 22, 33, 333

        inputs = torch.cat([img.to(device)]).unsqueeze(0)
        _, _, outputs = model(inputs)

        if isinstance(outputs, list):
            output = outputs[-1]
        else:
            output = outputs

        if cfg.TEST.FLIP_TEST:
            input_flipped = np.flip(inputs.cpu().numpy(), 3).copy()
            input_flipped = torch.from_numpy(input_flipped).cuda()
            _, _, outputs_flipped = model(input_flipped)

            if isinstance(outputs_flipped, list):
                output_flipped = outputs_flipped[-1]
            else:
                output_flipped = outputs_flipped

            output_flipped = flip_back(output_flipped.cpu().numpy(),
                                       dataset.flip_pairs)
            output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

            # feature is not aligned, shift flipped heatmap for higher accuracy
            # if config.TEST.SHIFT_HEATMAP:
            #     output_flipped[:, :, :, 1:] = \
            #         output_flipped.clone()[:, :, :, 0:-1]

            output = (output + output_flipped) * 0.5
        preds, maxvals = get_final_preds(
            cfg, output.clone().cpu().numpy(), None, None, transform_back=False)

        # del outputs
        # for h in feature_hooks:
        #     h.remove()
        # for h in atten_maps_hooks:
        #     h.remove()
        # for h in tokens_hooks:
        #     h.remove()

    # cache = get_local.cache
    # atten_weights = cache['Attention.forward'][:4]
    # layer1 = global_atten_maps[0].squeeze().detach().cpu().numpy()
    # l1_max = np.amax(layer1)
    # layer2 = global_atten_maps[1].squeeze().detach().cpu().numpy()
    # l2_max = np.amax(layer2)
    # file_path = 'kitpose2enc_w48_384_pre_coco_layer1_idx333.mat'
    # scipy.io.savemat(file_path, {'layer1': layer1})
    # file2_path = 'kitpose2enc_w48_384_pre_coco_layer2_idx333.mat'
    # scipy.io.savemat(file2_path, {'layer2': layer2})
    # visualize_layers(global_atten_maps, cols=2)

    # from heatmap_coord to original_image_coord
    query_locations = np.array([p * 4 + 0.5 for p in preds[0]])
    print(query_locations)
    inspect_feature_by_location(img, features[0], model, query_locations, threshold=0.0, idx=idx)


# activation = {}
#
#
# def get_activation(name):
#     def hook(model, input, output):
#         activation[name] = output.detach()
#
#     return hook

####################################### Visualize weight map ################################
# def main():
#     args = parse_args()
#     update_config(cfg, args)
#
#     logger, final_output_dir, tb_log_dir = create_logger(
#         cfg, args.cfg, 'visualization')
#
#     # cudnn related setting
#     cudnn.benchmark = cfg.CUDNN.BENCHMARK
#     torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
#     torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
#
#     device = torch.device('cuda:0')
#     model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
#         cfg, is_train=False
#     )
#     target_weight = eval('dataset.' + cfg.DATASET.DATASET + '.ReturnWeight')(cfg)
#
#     if cfg.TEST.MODEL_FILE:
#         logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
#         # for normal test
#         load_file = torch.load(cfg.TEST.MODEL_FILE)
#         model.load_state_dict(load_file, strict=False)
#     else:
#         logger.info('=> loading no model from ')
#     model.to(device)
#
#     # Data loading code
#     normalize = T.Normalize(
#         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#     )
#     dataset = eval('dataset.' + cfg.DATASET.DATASET)(
#         cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
#         T.Compose([
#             T.ToTensor(),
#             normalize,
#         ])
#     )
#
#     with torch.no_grad():
#         model.eval()
#         idx = 49
#         img = dataset[idx][0]  # dataset第一维是样本个数
#         target = dataset[idx][1]
#
#         inputs = torch.cat([img.to(device)]).unsqueeze(0)
#         outputs = model(inputs)
#
#         if isinstance(outputs, list):
#             output = outputs[-1]
#         else:
#             output = outputs
#
#         if cfg.TEST.FLIP_TEST:
#             input_flipped = np.flip(inputs.cpu().numpy(), 3).copy()
#             input_flipped = torch.from_numpy(input_flipped).cuda()
#             outputs_flipped = model(input_flipped)
#
#             if isinstance(outputs_flipped, list):
#                 output_flipped = outputs_flipped[-1]
#             else:
#                 output_flipped = outputs_flipped
#
#             output_flipped = transforms.flip_back(output_flipped.cpu().numpy(),
#                                                   dataset.flip_pairs)
#             output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
#
#             output = (output + output_flipped) * 0.5
#         preds, maxvals = get_final_preds(
#             cfg, output.clone().cpu().numpy(), None, None, transform_back=False)
#
#         query_location = np.array([p * 4 for p in preds[0]])
#
#         visualize_weight_map(inputs, output, target, target_weight, query_location, model_name='kitpose_w48_256',
#                              idx=idx,
#                              save_img=True)


####################################################################################################################

############################################## VISUALIZE CNN FEATURES ##########################################
# def main():
#     args = parse_args()
#     update_config(cfg, args)
#
#     logger, final_output_dir, tb_log_dir = create_logger(
#         cfg, args.cfg, 'visualization')
#
#     # cudnn related setting
#     cudnn.benchmark = cfg.CUDNN.BENCHMARK
#     torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
#     torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED
#
#     device = torch.device('cuda:0')
#     model = eval('models.' + cfg.MODEL.NAME + '.get_pose_net')(
#         cfg, is_train=False
#     )
#
#     if cfg.TEST.MODEL_FILE:
#         logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
#         # for normal test
#         load_file = torch.load(cfg.TEST.MODEL_FILE)
#         model.load_state_dict(load_file, strict=False)
#     else:
#         logger.info('=> loading no model from ')
#     model.to(device)
#
#     # Data loading code
#     normalize = T.Normalize(
#         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#     )
#     dataset = eval('dataset.' + cfg.DATASET.DATASET)(
#         cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
#         T.Compose([
#             T.ToTensor(),
#             normalize,
#         ])
#     )
#     # potentials = []
#     # features = []
#     # feature_hooks = [
#     #     model.pre_feature.final_layer.register_forward_hook(
#     #         lambda self, input, output1: features.append(output1)
#     #     )
#     # for i in range(cfg['MODEL']['NUM_JOINTS'])
#     # ]
#     # potentials_hooks = [
#     #     model.gm_relu.register_forward_hook(
#     #         lambda self, input, output: potentials.append(output)
#     #     )
#     #     # for i in range(cfg['MODEL']['NUM_JOINTS'])
#     # ]
#
#     with torch.no_grad():
#         model.eval()
#         idx = 21
#         img = dataset[idx][0]  # dataset第一维是样本个数
#         print(dataset.db[idx]['image'])
#         inputs = torch.cat([img.to(device)]).unsqueeze(0)
#         outputs = model(inputs)
#
#         if isinstance(outputs, list):
#             output = outputs[-1]
#         else:
#             output = outputs
#
#         # if cfg.TEST.FLIP_TEST:
#         #     input_flipped = np.flip(inputs.cpu().numpy(), 3).copy()
#         #     input_flipped = torch.from_numpy(input_flipped).cuda()
#         #     outputs_flipped = model(input_flipped)
#         #
#         #     if isinstance(outputs_flipped, list):
#         #         output_flipped = outputs_flipped[-1]
#         #     else:
#         #         output_flipped = outputs_flipped
#         #
#         #     output_flipped = transforms.flip_back(output_flipped.cpu().numpy(),
#         #                                           dataset.flip_pairs)
#         #     output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
#         #
#         #     output = (output + output_flipped) * 0.5
#         preds, maxvals = get_final_preds(
#             cfg, output.clone().cpu().numpy(), None, None, transform_back=False)
#
#         query_location = np.array([p * 4 for p in preds[0]])
#
#         visualize_activation_map(img, model, query_location, model_name='kitpose2enc_w32_256_new', threshold=1e-3,
#                                  save_img=True, idx=idx)


################################################################################################################
#
#         # del outputs
#         # for h in feature_hooks:
#         #     h.remove()
#         # for h in potentials_hooks:
#         #     h.remove()
#
#     # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 8))
#     # std = torch.tensor([0.229, 0.224, 0.225]).unsqueeze(-1).unsqueeze(-1)
#     # mean = torch.tensor([0.485, 0.456, 0.406]).unsqueeze(-1).unsqueeze(-1)
#     # img = img * std + mean
#     # img = img.permute(1, 2, 0).detach().cpu().numpy()
#     # for i in range(cfg['MODEL']['NUM_JOINTS']):
#     #     feature = features[0]
#     #     potential = potentials[i]
#     #     feature = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)(feature)
#     #     potential = torch.nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)(potential)
#     #     feature = feature.squeeze()[i].detach().cpu().numpy()
#     #     potential = potential.squeeze().detach().cpu().numpy()
#     #     ###########
#     #     ax1.imshow(img)
#     #     ax2.imshow(img)
#     #     feature = feature[..., np.newaxis]
#     #     potential = potential[..., np.newaxis]
#     #     feature_mask = feature <= 0.0
#     #     feature[feature_mask] = 0
#     #
#     #     potential_mask = potential <= 0.0
#     #     potential[potential_mask] = 0
#     #     ft = ax1.imshow(feature, cmap="nipy_spectral", alpha=0.79)
#     #     pt = ax2.imshow(potential, cmap="nipy_spectral", alpha=0.79)
#     #     # ax1.imshow(feature)
#     #     # ax2.imshow(potential)
#     #     # plt.show()
#     #     ############
#     # plt.show()


##########################################################################

# cache = get_local.cache


if __name__ == '__main__':
    main()
