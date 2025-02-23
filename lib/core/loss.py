# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from utils.kpt_info import keypoint_info
from torch import linalg as LA


class BPLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.l2_criterion = nn.PairwiseDistance(eps=1e-6, keepdim=True)
        self.mse_criterion = nn.MSELoss(reduction='mean')
        self.m = margin
        self.eps = 1e-7

    def forward(self, x):
        b, num_parts, h, w = x.shape
        # num_joints = target.size(1)
        # heatmap_gt = target.reshape((b, num_joints, -1))
        # heatmap_gt = torch.sum(heatmap_gt, dim=1)

        if x.ndim == 4:
            x = x.reshape((b, num_parts, -1))
        hidden_dim = h * w
        heatmap_part = torch.sum(x, dim=1)
        # mse_loss = 0.5 * self.mse_criterion(heatmap_part, heatmap_gt)

        part_loss = 0
        # part_loss_item = 0
        cindices = list(itertools.combinations(range(num_parts), 2))

        for cindex in cindices:
            part_dist = self.l2_criterion(x[:, cindex[0]], x[:, cindex[1]]) / hidden_dim
            # part_loss.append(part_loss_item)
            # dist = torch.stack(part_loss, dim=0).mean(dim=0)
            part_delta = self.m - part_dist
            part_delta = torch.clamp(part_delta, min=0.0, max=None)
            part_delta = torch.mean(part_delta.pow(2))
            part_loss += 0.5 * part_delta

        return part_loss / len(cindices)
        # t = x.unsqueeze(1).repeat(1, num_parts, 1, 1)
        # s = x.unsqueeze(2).repeat(1, 1, num_parts, 1)
        # # dist = torch.square(t - s).mean(dim=-1)
        # # 2023/08/13 update
        # # dist = (t - s).pow(2).sum(-1)
        # res = self.criterion(t, s).mean(dim=-1)
        # mask = torch.ones((B, num_parts, num_parts), device=x.device)
        # mask = torch.triu(mask, diagonal=1)
        # # mask =
        # # dist = mask * dist
        # # delta = self.m - dist
        # # delta = delta * mask
        # # delta = torch.clamp(delta, min=0.0, max=None)
        # # delta = torch.pow(delta, 2)
        # res = mask * res
        # res = 1.0 / 1e-2 + torch.exp(res)
        # return res.mean()
        # # return delta.pow(2).mean()


# class BPLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super().__init__()
#         self.criterion = nn.MSELoss(reduction='none')
#         self.m = margin
#
#     def forward(self, x):
#         B, num_parts = x.size(0), x.size(1)
#         if x.ndim == 4:
#             x = x.reshape((B, num_parts, -1))
#
#         t = x.unsqueeze(1).repeat(1, num_parts, 1, 1)
#         s = x.unsqueeze(2).repeat(1, 1, num_parts, 1)
#         dist = torch.square(t - s).mean(dim=-1)
#         # dist = self.criterion(t, s).mean(dim=-1)
#         mask = torch.ones((B, num_parts, num_parts), device=x.device)
#         mask = torch.triu(mask, diagonal=1)
#         # mask =
#         res = mask * dist
#         # delta = self.m - dist
#         # delta = torch.clamp(delta, min=0.0, max=None)
#         # delta = torch.pow(delta, 2)
#         res = 1.0 / 1e-2 + torch.exp(-1.0 * res)
#         return res.mean()


class QFL(nn.Module):
    def __init__(self, beta=2):
        super().__init__()
        self.beta = beta

    def forward(self, pred, target, target_weight):
        batch_size = pred.size(0)
        num_joints = pred.size(1)

        heatmaps_pred = pred.reshape((batch_size, num_joints, -1))
        heatmaps_gt = target.reshape((batch_size, num_joints, -1))
        err_item = torch.abs(heatmaps_pred - heatmaps_gt)
        weight = torch.abs(1 - heatmaps_pred) * heatmaps_gt ** 0.01 + torch.abs(heatmaps_pred) * (
                1 - heatmaps_gt ** 0.01)
        loss = F.binary_cross_entropy_with_logits(heatmaps_pred, heatmaps_gt, reduction='none') * err_item.pow(
            self.beta) * weight
        return loss.mean()


class GFL(nn.Module):
    def __init__(self, beta=2):
        super().__init__()
        self.beta = beta
        self.smooth_blur = T.GaussianBlur(kernel_size=13, sigma=4)
        self.sharp_kernel = torch.nn.Parameter(
            torch.tensor(
                [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
                dtype=torch.float32,
            ).unsqueeze(0).unsqueeze(0),
            requires_grad=False
        )

        self.kl_loss = nn.KLDivLoss(reduction='none', log_target=True)
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, pred, target, target_weight):
        batch_size = pred.size(0)
        num_joints = pred.size(1)
        loss = []

        heatmaps_gt = target.split(1, 1)
        heatmaps_pred = pred.reshape((batch_size, num_joints, -1)).split(1, 1)

        for idx in range(num_joints):
            heatmap_gt = heatmaps_gt[idx]
            heatmap_pred = heatmaps_pred[idx]
            smooth_gt = self.smooth_blur(heatmap_gt).reshape(batch_size, 1, -1)
            sharp_gt = F.conv2d(heatmap_gt, self.sharp_kernel, padding=1).reshape(batch_size, 1, -1)
            heatmap_gt = heatmap_gt.reshape(batch_size, 1, -1)
            wl = torch.abs(sharp_gt - heatmap_gt)
            wr = torch.abs(heatmap_gt - smooth_gt)
            # wl = torch.norm(sharp_gt - heatmap_pred)
            # wr = torch.norm(heatmap_pred - smooth_gt)
            err_item = (heatmap_pred - heatmap_gt) ** self.beta
            # heatmap_pred = F.log_softmax(heatmap_pred, dim=-1)
            loss_item = self.mse_loss(heatmap_pred, sharp_gt) * wl + self.mse_loss(heatmap_pred, smooth_gt) * wr
            loss_item = loss_item * err_item
            loss.append(loss_item)

        loss = torch.cat(loss, dim=1)

        return loss.mean()


class GFL_5x5(nn.Module):
    def __init__(self, beta=2):
        super().__init__()
        self.beta = beta
        self.smooth_blur = T.GaussianBlur(kernel_size=13, sigma=4)
        self.sharp_kernel = torch.nn.Parameter(
            0.25 * torch.tensor(
                [
                    [0, 0, -1, 0, 0],
                    [0, -1, -2, -1, 0],
                    [-1, -2, 16, -2, -1],
                    [0, -1, -2, -1, 0],
                    [0, 0, -1, 0, 0]
                ],
                dtype=torch.float32,
            ).unsqueeze(0).unsqueeze(0),
            requires_grad=False
        )
        self.kl_loss = nn.KLDivLoss(reduction='none', log_target=True)
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, pred, target, target_weight):
        batch_size = pred.size(0)
        num_joints = pred.size(1)
        loss = []

        heatmaps_gt = target.split(1, 1)
        heatmaps_pred = pred.reshape((batch_size, num_joints, -1)).split(1, 1)

        for idx in range(num_joints):
            heatmap_gt = heatmaps_gt[idx]
            heatmap_pred = heatmaps_pred[idx]
            smooth_gt = self.smooth_blur(heatmap_gt).reshape(batch_size, 1, -1)
            sharp_gt = F.conv2d(heatmap_gt, self.sharp_kernel, padding=2).reshape(batch_size, 1, -1)
            heatmap_gt = heatmap_gt.reshape(batch_size, 1, -1)
            wl = torch.abs(sharp_gt - heatmap_gt)
            wr = torch.abs(heatmap_gt - smooth_gt)
            # wl = torch.norm(sharp_gt - heatmap_pred)
            # wr = torch.norm(heatmap_pred - smooth_gt)
            err_item = (heatmap_pred - heatmap_gt) ** self.beta
            # heatmap_pred = F.log_softmax(heatmap_pred, dim=-1)
            loss_item = self.mse_loss(heatmap_pred, sharp_gt) * wl + self.mse_loss(heatmap_pred, smooth_gt) * wr
            loss_item = loss_item * err_item
            loss.append(loss_item)

        loss = torch.cat(loss, dim=1)

        return loss.mean()


class GFL_7x7(nn.Module):
    def __init__(self, beta=2):
        super().__init__()
        self.beta = beta
        self.smooth_blur = T.GaussianBlur(kernel_size=13, sigma=4)
        self.sharp_kernel = torch.nn.Parameter(
            torch.tensor(
                [
                    [0, 0, -1, -1, -1, 0, 0],
                    [0, -1, -3, -3, -3, -1, 0],
                    [-1, -3, 0, 7, 0, -3, -1],
                    [-1, -3, 7, 24, 7, -3, -1],
                    [-1, -3, 0, 7, 0, -3, -1],
                    [0, -1, -3, -3, -3, -1, 0],
                    [0, 0, -1, -1, -1, 0, 0],
                ],
                dtype=torch.float32,
            ).unsqueeze(0).unsqueeze(0) * 1.0 / 6.0,
            requires_grad=False
        )
        self.kl_loss = nn.KLDivLoss(reduction='none', log_target=True)
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, pred, target, target_weight):
        batch_size = pred.size(0)
        num_joints = pred.size(1)
        loss = []

        heatmaps_gt = target.split(1, 1)
        heatmaps_pred = pred.reshape((batch_size, num_joints, -1)).split(1, 1)

        for idx in range(num_joints):
            heatmap_gt = heatmaps_gt[idx]
            heatmap_pred = heatmaps_pred[idx]
            smooth_gt = self.smooth_blur(heatmap_gt).reshape(batch_size, 1, -1)
            sharp_gt = F.conv2d(heatmap_gt, self.sharp_kernel, padding=3).reshape(batch_size, 1, -1)
            heatmap_gt = heatmap_gt.reshape(batch_size, 1, -1)
            wl = torch.abs(sharp_gt - heatmap_gt)
            wr = torch.abs(heatmap_gt - smooth_gt)
            # wl = torch.norm(sharp_gt - heatmap_pred)
            # wr = torch.norm(heatmap_pred - smooth_gt)
            err_item = (heatmap_pred - heatmap_gt) ** self.beta
            # heatmap_pred = F.log_softmax(heatmap_pred, dim=-1)
            loss_item = self.mse_loss(heatmap_pred, sharp_gt) * wl + self.mse_loss(heatmap_pred, smooth_gt) * wr
            loss_item = loss_item * err_item
            loss.append(loss_item)

        loss = torch.cat(loss, dim=1)

        return loss.mean()


# class StructureLoss(nn.Module):
#     '''
#     body part heatmap损失函数
#     '''
#
#     def __init__(self, use_target_weight, dataset='ap10k'):
#         super(StructureLoss, self).__init__()
#         self.criterion = nn.MSELoss(reduction='mean')
#         self.use_target_weight = use_target_weight
#         self.kpt_names, self.kpt_dep, self.kpt_dict = keypoint_info(dataset)
#
#     def forward(self, output, target, target_weight, joints_weight):
#         batch_size = output.size(0)
#         num_joints = output.size(1)
#         reg_lambda = 1e-3
#         s_lambda = 0.5
#         heatmaps_pred = output.reshape((batch_size, num_joints, -1))
#         heatmaps_gt = target.reshape((batch_size, num_joints, -1))
#         loss = 0
#
#         for idx, kpt_name in enumerate(self.kpt_names):
#             heatmap_pred = heatmaps_pred[:, idx].squeeze()
#             heatmap_gt = heatmaps_gt[:, idx].squeeze()
#             if self.use_target_weight:
#                 loss += 0.5 * self.criterion(
#                     heatmap_pred.mul(target_weight[:, idx]),
#                     heatmap_gt.mul(target_weight[:, idx])
#                 )
#             else:
#                 loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
#
#             structrue_list = []
#             for cond_kpt in self.kpt_dep[kpt_name]:
#                 cond_kpt_id = self.kpt_dict[cond_kpt]
#                 structrue_list.append(cond_kpt_id)
#             structure_heatmap_pred = heatmaps_pred[:, structrue_list].sum(dim=1)
#             structure_heatmap_gt = heatmaps_gt[:, structrue_list].sum(dim=1)
#             loss += 0.5 * s_lambda * self.criterion(structure_heatmap_pred, structure_heatmap_gt)
#
#         reg = torch.square(LA.norm(joints_weight - 1.))
#
#         return loss / num_joints + reg_lambda * reg

class StructureLoss(nn.Module):
    '''
    body part heatmap损失函数
    '''

    def __init__(self, use_target_weight, dataset='ap10k'):
        super(StructureLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.kpt_names, self.kpt_dep, self.kpt_dict = keypoint_info(dataset)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean', log_target=True)

    def forward(self, output, target, target_weight, joints_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        reg_lambda = 1e-3
        s_lambda = 0.5
        heatmaps_pred = output.reshape((batch_size, num_joints, -1))
        heatmaps_gt = target.reshape((batch_size, num_joints, -1))
        loss = 0

        for idx, kpt_name in enumerate(self.kpt_names):
            heatmap_pred = heatmaps_pred[:, idx].squeeze()
            heatmap_gt = heatmaps_gt[:, idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

            structrue_list = []
            for cond_kpt in self.kpt_dep[kpt_name]:
                cond_kpt_id = self.kpt_dict[cond_kpt]
                structrue_list.append(cond_kpt_id)
            structure_heatmap_pred = heatmaps_pred[:, structrue_list].sum(dim=1)
            structure_heatmap_pred = F.log_softmax(structure_heatmap_pred, dim=-1)
            structure_heatmap_gt = heatmaps_gt[:, structrue_list].sum(dim=1)
            structure_heatmap_gt = F.log_softmax(structure_heatmap_gt, dim=-1)
            loss += 0.5 * s_lambda * self.kl_loss(structure_heatmap_pred, structure_heatmap_gt)

        reg = torch.square(LA.norm(joints_weight - 1.))

        return loss / num_joints + reg_lambda * reg


# class JointsMSELoss(nn.Module):
#     '''
#     heatmap的损失函数
#     Args：
#         - use_target_weight: 不同joints有不同的weight
#     '''
#
#     def __init__(self, use_target_weight):
#         super(JointsMSELoss, self).__init__()
#         self.criterion = nn.MSELoss(reduction='mean')
#         self.use_target_weight = use_target_weight
#
#     def forward(self, output, target, target_weight, joints_weight=None):
#         batch_size = output.size(0)
#         num_joints = output.size(1)
#         reg_lambda = 1e-2
#         heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
#         # 在dim=1上进行split，tuple=num_joints,每一个元素大小维为(batch,1,4096)
#         heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
#         loss = 0
#
#         for idx in range(num_joints):
#             heatmap_pred = heatmaps_pred[idx].squeeze()
#             heatmap_gt = heatmaps_gt[idx].squeeze()
#             if self.use_target_weight:
#                 loss += 0.5 * self.criterion(
#                     heatmap_pred.mul(target_weight[:, idx]),
#                     heatmap_gt.mul(target_weight[:, idx])
#                 )
#             else:
#                 loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)
#         reg = torch.norm(joints_weight - 1., 1)
#         # reg = torch.square(LA.norm(joints_weight - 1.))
#         # return loss / num_joints
#         return loss / num_joints + reg_lambda * reg
class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class SmoothJointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(SmoothJointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class HeatmapLoss(nn.Module):
    def __init__(self, use_target_weight=True, beta=1):
        super(HeatmapLoss, self).__init__()
        # self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.beta = beta

    def forward(self, output, target, target_weight, joints_weight=None):
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape((batch_size, num_joints, -1))
        heatmaps_gt = target.reshape((batch_size, num_joints, -1))

        # weight = torch.abs(1 - heatmaps_pred) * heatmaps_gt ** 0.01 + torch.abs(heatmaps_pred) * (
        #         1 - heatmaps_gt ** 0.01)
        weight = torch.abs(heatmaps_pred - heatmaps_gt)

        # loss = 0.5 * (heatmaps_pred - heatmaps_gt) ** 2 * weight * target_weight

        # loss = 0.5 * (heatmaps_pred - heatmaps_gt) ** 2 * weight
        loss = weight.pow(self.beta) * target_weight * (heatmaps_pred - heatmaps_gt) ** 2

        loss = loss.mean()

        return loss


class AdaptiveLoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(AdaptiveLoss, self).__init__()
        # self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight, joints_weight=None):
        batch_size = output.size(0)
        num_joints = output.size(1)

        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_gt = heatmaps_gt[idx].squeeze()
            heatmap_pred = heatmaps_pred[idx].squeeze()
            weight = torch.abs(1 - heatmap_pred) * heatmap_gt ** 0.01 + torch.abs(heatmap_pred) * (
                    1 - heatmap_gt ** 0.01)
            loss += 0.5 * (heatmap_pred - heatmap_gt) ** 2 * weight
        loss = loss / num_joints

        return loss.mean()


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)
