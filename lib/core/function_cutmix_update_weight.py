# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Hanbin Dai (daihanbin.ac@gmail.com) and Feng Zhang (zhangfengwcy@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os

import numpy as np
import torch

from core.evaluate import accuracy
from core.inference import get_final_preds
from core.inference import get_max_preds
from utils.transforms import flip_back
from utils.vis import save_debug_images
from core.function import AverageMeter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------------
def cutmix_data_no_target_weight_update(input, target, target_weight, alpha=1.0, cutmix_prob=1.0):
    ''' Returns mixed inputs, pairs of targets, and lambda'''

    r = np.random.rand(1)
    if alpha > 0 and r < cutmix_prob:
        lam = np.random.beta(alpha * 2, alpha)
    else:
        lam = -1
        return input, target, target_weight, None, lam

    batch_size = input.size()[0]
    index = torch.randperm(batch_size).cuda()

    bbx1, bby1, bbx2, bby2 = rand_bbox(input.size(), lam)
    bbx1_t, bby1_t, bbx2_t, bby2_t = bbx1 // 4, bby1 // 4, bbx2 // 4, bby2 // 4

    # -------- create new tensors----------
    input_A = input.clone()
    target_A = target.clone()
    target_weight_A = target_weight.clone().view(-1, target.size(1))

    input_B = input[index].clone()
    target_B = target[index].clone()
    target_weight_B = target_weight[index].clone().view(-1, target.size(1))

    input_mix = input_A
    input_mix[:, :, bbx1:bbx2, bby1:bby2] = input_B[:, :, bbx1:bbx2, bby1:bby2]

    target_mix = target_A
    target_mix[:, :, bbx1_t:bbx2_t, bby1_t:bby2_t] = target_B[:, :, bbx1_t:bbx2_t, bby1_t:bby2_t]

    target_weight_mix_A = target_weight_A.view(-1, target.size(1), 1)
    target_weight_mix_B = target_weight_B.view(-1, target.size(1), 1)

    # -----------------------------------------------------
    # reweight based on bounding box size.
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (input.size()[-1] * input.size()[-2])

    # -----------------------------------------------------
    return input_mix, target_mix, target_weight_mix_A, target_weight_mix_B, lam


# ---------------------------------------------------------------------------------

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1 - lam)
    # cut_rat = 1 - lam
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix_criterion(criterion, pred, target, tweighta, tweightb, lam):
    return lam * criterion(pred, target, tweighta) + (1 - lam) * criterion(pred, target, tweightb)


def train_cutmix(config, train_loader, model, criterion, optimizer, epoch, joints_weight,
                 output_dir, tb_log_dir, wandb=None, writer_dict=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, target_weight, meta) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        # m_input, m_target, m_tweia, m_tweib, lam = cutmix_data(input, target, target_weight, alpha=1.0, cutmix_prob=1.0)
        m_input, m_target, m_tweia, m_tweib, lam = \
            cutmix_data_no_target_weight_update(input, target, target_weight, alpha=1.0, cutmix_prob=1.0)

        # compute output
        outputs = model(m_input)

        m_target = m_target.cuda(non_blocking=True)
        m_tweia = m_tweia.cuda(non_blocking=True)
        if m_tweib is not None:
            m_tweib = m_tweib.cuda(non_blocking=True)

        if isinstance(outputs, list):
            # loss = criterion(outputs[0], target, target_weight)
            loss = cutmix_criterion(criterion, outputs[0], m_target, m_tweia, m_tweib, lam)
            for output in outputs[1:]:
                # loss += criterion(output, target, target_weight)
                loss += cutmix_criterion(criterion, output, m_target, m_tweia, m_tweib, lam)
        elif not isinstance(outputs, list) and m_tweib is not None:
            output = outputs
            # loss = criterion(output, target, target_weight)
            loss = cutmix_criterion(criterion, output, m_target, m_tweia, m_tweib, lam)
        elif not isinstance(outputs, list) and m_tweib is None:
            output = outputs
            loss = criterion(output, m_target, m_tweia)
        else:
            loss = criterion(outputs[0], m_target, m_tweia)
            for output in outputs[1:]:
                loss += criterion(output, m_target, m_tweia)
        # loss = criterion(output, target, target_weight)

        # compute gradient and do update step
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                         target.detach().cpu().numpy())
        acc.update(avg_acc, cnt)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t' \
                  'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                speed=input.size(0) / batch_time.val,
                data_time=data_time, loss=losses, acc=acc)
            logger.info(msg)

            writer = writer_dict['writer']
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', losses.val, global_steps)
            writer.add_scalar('train_acc', acc.val, global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            prefix = '{}_{}'.format(os.path.join(output_dir, 'train'), i)
            save_debug_images(config, m_input, meta, m_target, pred * 4, output,
                              prefix)
    wandb_train_log = {
        "train_loss": losses.val,
        "train_acc": acc.val,
    }
    return wandb_train_log


def validate(config, val_loader, val_dataset, model, criterion, output_dir, joints_weight,
             tb_log_dir, wandb=None, writer_dict=None, log_type='map'):
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    # switch to evaluate mode
    model.eval()

    num_samples = len(val_dataset)
    all_preds = np.zeros(
        (num_samples, config.MODEL.NUM_JOINTS, 3),
        dtype=np.float32
    )
    all_boxes = np.zeros((num_samples, 6))
    image_path = []
    filenames = []
    imgnums = []
    bbox_ids = []
    idx = 0
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute output
            outputs = model(input)
            if isinstance(outputs, list):
                output = outputs[-1]
            else:
                output = outputs

            if config.TEST.FLIP_TEST:
                # this part is ugly, because pytorch has not supported negative index
                # input_flipped = model(input[:, :, :, ::-1])
                input_flipped = np.flip(input.cpu().numpy(), 3).copy()
                input_flipped = torch.from_numpy(input_flipped).cuda()
                outputs_flipped = model(input_flipped)

                if isinstance(outputs_flipped, list):
                    output_flipped = outputs_flipped[-1]
                else:
                    output_flipped = outputs_flipped

                output_flipped = flip_back(output_flipped.cpu().numpy(),
                                           val_dataset.flip_pairs)
                output_flipped = torch.from_numpy(output_flipped.copy()).cuda()

                # feature is not aligned, shift flipped heatmap for higher accuracy
                # if config.TEST.SHIFT_HEATMAP:
                #     output_flipped[:, :, :, 1:] = \
                #         output_flipped.clone()[:, :, :, 0:-1]

                output = (output + output_flipped) * 0.5

            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            # if config.LOSS.USE_DIFFERENT_JOINTS_WEIGHT:
            #     target_weight = torch.mul(target_weight[:], joints_weight)

            loss = criterion(output, target, target_weight)

            num_images = input.size(0)
            # measure accuracy and record loss
            losses.update(loss.item(), num_images)
            _, avg_acc, cnt, pred = accuracy(output.cpu().numpy(),
                                             target.cpu().numpy())

            acc.update(avg_acc, cnt)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            c = meta['center'].numpy()
            s = meta['scale'].numpy()
            score = meta['bbox_score'].numpy()

            if bbox_ids is not None:
                bbox_ids.append(meta['bbox_id'])

            preds, maxvals = get_final_preds(
                config, output.clone().cpu().numpy(), c, s)

            all_preds[idx:idx + num_images, :, 0:2] = preds[:, :, 0:2]
            all_preds[idx:idx + num_images, :, 2:3] = maxvals
            # double check this all_boxes parts
            all_boxes[idx:idx + num_images, 0:2] = c[:, 0:2]
            all_boxes[idx:idx + num_images, 2:4] = s[:, 0:2]
            all_boxes[idx:idx + num_images, 4] = np.prod(s * 200, 1)
            all_boxes[idx:idx + num_images, 5] = score
            image_path.extend(meta['image'])

            idx += num_images

            if i % config.PRINT_FREQ == 0:
                msg = 'Test: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t' \
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t' \
                      'Accuracy {acc.val:.3f} ({acc.avg:.3f})'.format(
                    i, len(val_loader), batch_time=batch_time,
                    loss=losses, acc=acc)
                logger.info(msg)

                prefix = '{}_{}'.format(
                    os.path.join(output_dir, 'val'), i
                )
                save_debug_images(config, input, meta, target, pred * 4, output,
                                  prefix)

        bbox_ids = torch.cat(bbox_ids)
        name_values, perf_indicator = val_dataset.evaluate(
            config, all_preds, output_dir, all_boxes, bbox_ids, image_path,
            filenames, imgnums
        )

        # wandb log setting
        if wandb is not None and log_type == 'map':
            wandb_val_log = {
                "val_loss": losses.avg,
                "val_acc": acc.avg,
                "mAP": perf_indicator,
                "AP.5": name_values['Ap .5'],
                "AP.75": name_values['AP .75'],
                "AP(M)": name_values['AP (M)'],
                "AP(L)": name_values['AP (L)'],
                "mAR": name_values['AR'],
                "AR.5": name_values['AR .5'],
                "AR.75": name_values['AR .75'],
                "AR(M)": name_values['AR (M)'],
                "AR(L)": name_values['AR (L)'],
            }
        elif wandb is not None and log_type == 'pck':
            wandb_val_log = {
                "val_loss": losses.avg,
                "val_acc": acc.avg,
                "Head": name_values['Head'],
                "Shoulder": name_values['Shoulder'],
                "Elbow": name_values['Elbow'],
                "Wrist": name_values['Wrist'],
                "Hip": name_values['Hip'],
                "Knee": name_values['Knee'],
                "Ankle": name_values['Ankle'],
                "Mouth": name_values['Mouth'],
                "Tail": name_values['Tail'],
                "Mean": name_values['Mean'],
            }
        else:
            wandb_val_log = {}

        model_name = config.MODEL.NAME
        if isinstance(name_values, list):
            for name_value in name_values:
                _print_name_value(name_value, model_name)
        else:
            _print_name_value(name_values, model_name)

        if writer_dict:
            writer = writer_dict['writer']
            global_steps = writer_dict['valid_global_steps']
            writer.add_scalar(
                'valid_loss',
                losses.avg,
                global_steps
            )
            writer.add_scalar(
                'valid_acc',
                acc.avg,
                global_steps
            )
            if isinstance(name_values, list):
                for name_value in name_values:
                    writer.add_scalars(
                        'valid',
                        dict(name_value),
                        global_steps
                    )
            else:
                writer.add_scalars(
                    'valid',
                    dict(name_values),
                    global_steps
                )
            writer_dict['valid_global_steps'] = global_steps + 1

    return perf_indicator, wandb_val_log


# markdown format output
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


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0
