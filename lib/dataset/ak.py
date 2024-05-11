# # ------------------------------------------------------------------
# # Copyright (c) Microsoft
# # Licensed under the MIT license.
# # Written by Bin XIao (Bin.Xiao@microsoft.com)
# # ----------------------------------------------------------------
#
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import math
# import logging
# import os
# import json_tricks as json
# from collections import OrderedDict
# import json
# import numpy as np
# import torch
# from scipy.io import loadmat, savemat
#
# from dataset.JointsDataset import JointsDataset
#
# logger = logging.getLogger(__name__)
#
#
# class AnimalKingdomDataset(JointsDataset):
#     def __init__(self, cfg, root, image_set, is_train, transform=None):
#         super().__init__(cfg, root, image_set, is_train, transform)
#         '''
#         0: Head_Mid_Top
#         1: Eye_Left
#         2: Eye_Right
#         3: Mouth_Front_Top
#         4: Mouth_Back_Left
#         5: Mouth_Back_Right
#         6: Mouth_Front_Bottom
#         7: Shoulder_Left
#         8: Shoulder_Right
#         9: Elbow_Left
#         10: Elbow_Right
#         11: Wrist_Left
#         12: Wrist_Right
#         13: Torso_Mid_Back
#         14: Hip_Left
#         15: Hip_Right
#         16: Knee_Left
#         17: Knee_Right
#         18: Ankle_Left
#         19: Ankle_Right
#         20: Tail_Top_Back
#         21: Tail_Mid_Back
#         22: Tail_End_Back
#         '''
#         self.num_joints = 23
#         self.flip_pairs = [[1, 2], [4, 5], [7, 8], [9, 10], [11, 12], [14, 15], [16, 17], [18, 19]]
#         self.parent_ids = [3, 3, 3, 6, 6, 6, 13, 13, 13, 7, 8, 9, 10, 13, 13, 13, 14, 15, 16, 17, 12, 20, 21]
#         self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
#         self.lower_body_ids = (14, 15, 16, 17, 18, 19, 20, 21, 22)
#
#         self.db = self._get_db()
#
#         if is_train and cfg.DATASET.SELECT_DATA:
#             self.db = self.select_data(self.db)
#
#         logger.info('==> {} samples'.format(len(self.db)))
#
#     def _get_db(self):
#         # Create train/val split
#         file_name = os.path.join(
#             self.root, 'annot', self.image_set + '.json'
#         )
#         with open(file_name) as anno_file:
#             anno = json.load(anno_file)
#
#         gt_db = []
#         bbox_id = 0
#         for a in anno:
#             image_name = a['image']
#
#             c = np.array(a['center'], dtype=np.float)
#             s = np.array([a['scale'], a['scale']], dtype=np.float)
#
#             # Adjust center/scale slightly to avoid cropping limbs
#             if c[0] != -1:
#                 # c[1] = c[1] + 15 * s[1]
#                 s = s * 1.25
#
#             # MPII uses matlab format, index is based 1, we should first convert to 0-based index
#             # c = c - 1
#             joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
#             joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
#             if self.image_set != 'test':
#                 joints = np.array(a['joints'])
#                 joints[:, 0:2] = joints[:, 0:2]
#                 joints_vis = np.array(a['joints_vis'])
#                 assert len(joints) == self.num_joints, 'joint num diff: {} vs {}'.format(len(joints), self.num_joints)
#
#                 joints_3d[:, 0:2] = joints[:, 0:2]
#                 joints_3d_vis[:, 0] = joints_vis[:]
#                 joints_3d_vis[:, 1] = joints_vis[:]
#
#             image_dir = 'image.zip@' if self.data_format == 'zip' else 'images'
#             gt_db.append(
#                 {
#                     'image': os.path.join(self.root, image_dir, image_name),
#                     'center': c,
#                     'scale': s,
#                     'joints_3d': joints_3d,
#                     'joints_3d_vis': joints_3d_vis,
#                     'filename': '',
#                     'imgnum': 0,
#                     'bbox_score': 1,
#                     'bbox_id': bbox_id,
#                 }
#             )
#             bbox_id = bbox_id + 1
#         return gt_db
#
#     def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
#         # Convert 0-based index to 1-based index
#         preds = preds[:, :, 0:2] + 1.0
#         if output_dir:
#             pred_file = os.path.join(output_dir, 'pred.mat')
#             savemat(pred_file, mdict={'preds': preds})
#
#         # if 'test' in cfg.DATASET.TEST_SET:
#         #     return {'Null': 0.0}, 0.0
#
#         SC_BIAS = 1
#         threshold = 0.05
#         gt_file = os.path.join(cfg.DATASET.ROOT,
#                                'annot',
#                                '{}.json'  # gt_{}.json
#                                .format(cfg.DATASET.TEST_SET)
#                                )
#
#         with open(gt_file) as f:
#             gt_dict = json.load(f)
#
#         ### Changes below ###
#         # dataset_joints = t_dict['dataset_joints']
#         # jnt_visible = [v for k, v in gt_dict['joints_vis'].items()]
#         # pos_gt_src = [v for k, v in gt_dict['joints'].items()]
#         # scale = [v for k, v in gt_dict['scale'].items()]
#
#         dataset_joints = [
#             [
#                 "Head_Mid_Top"
#             ],
#             [
#                 "Eye_Left"
#             ],
#             [
#                 "Eye_Right"
#             ],
#             [
#                 "Mouth_Front_Top"
#             ],
#             [
#                 "Mouth_Back_Left"
#             ],
#             [
#                 "Mouth_Back_Right"
#             ],
#             [
#                 "Mouth_Front_Bottom"
#             ],
#             [
#                 "Shoulder_Left"
#             ],
#             [
#                 "Shoulder_Right"
#             ],
#             [
#                 "Elbow_Left"
#             ],
#             [
#                 "Elbow_Right"
#             ],
#             [
#                 "Wrist_Left"
#             ],
#             [
#                 "Wrist_Right"
#             ],
#             [
#                 "Torso_Mid_Back"
#             ],
#             [
#                 "Hip_Left"
#             ],
#             [
#                 "Hip_Right"
#             ],
#             [
#                 "Knee_Left"
#             ],
#             [
#                 "Knee_Right"
#             ],
#             [
#                 "Ankle_Left"
#             ],
#             [
#                 "Ankle_Right"
#             ],
#             [
#                 "Tail_Top_Back"
#             ],
#             [
#                 "Tail_Mid_Back"
#             ],
#             [
#                 "Tail_End_Back"
#             ]
#         ]
#
#         jnt_visible = [x['joints_vis'] for x in gt_dict]
#         pos_gt_src = [x['joints'] for x in gt_dict]
#         scale = [x['scale'] for x in gt_dict]
#
#         scale = np.array(scale)
#         scale = scale * 200 * math.sqrt(2)
#
#         jnt_visible = np.transpose(jnt_visible, [1, 0])
#         pos_pred_src = np.transpose(preds, [1, 2, 0])
#         pos_gt_src = np.transpose(pos_gt_src, [1, 2, 0])
#         dataset_joints = np.array(dataset_joints)
#         # np.where(dataset_joints == 'Shoulder_Left') = ([7], [0])
#         head = np.where(dataset_joints == 'Head_Mid_Top')[0][0]
#         lsho = np.where(dataset_joints == 'Shoulder_Left')[0][0]
#         lelb = np.where(dataset_joints == 'Elbow_Left')[0][0]
#         lwri = np.where(dataset_joints == 'Wrist_Left')[0][0]
#         lhip = np.where(dataset_joints == 'Hip_Left')[0][0]
#         lkne = np.where(dataset_joints == 'Knee_Left')[0][0]
#         lank = np.where(dataset_joints == 'Ankle_Left')[0][0]
#
#         rsho = np.where(dataset_joints == 'Shoulder_Right')[0][0]
#         relb = np.where(dataset_joints == 'Elbow_Right')[0][0]
#         rwri = np.where(dataset_joints == 'Wrist_Right')[0][0]
#         rkne = np.where(dataset_joints == 'Knee_Right')[0][0]
#         rank = np.where(dataset_joints == 'Ankle_Right')[0][0]
#         rhip = np.where(dataset_joints == 'Hip_Right')[0][0]
#
#         tmouth = np.where(dataset_joints == 'Mouth_Front_Top')[0][0]
#         lmouth = np.where(dataset_joints == 'Mouth_Back_Left')[0][0]
#         rmouth = np.where(dataset_joints == 'Mouth_Back_Right')[0][0]
#         bmouth = np.where(dataset_joints == 'Mouth_Front_Bottom')[0][0]
#         ttail = np.where(dataset_joints == 'Tail_Top_Back')[0][0]
#         mtail = np.where(dataset_joints == 'Tail_Mid_Back')[0][0]
#         btail = np.where(dataset_joints == 'Tail_End_Back')[0][0]
#
#         uv_error = pos_pred_src - pos_gt_src
#
#         uv_err = np.linalg.norm(uv_error, axis=1)
#
#         scale *= SC_BIAS
#         headsizes = scale
#
#         scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
#         scaled_uv_err = np.divide(uv_err, scale)
#         scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
#         jnt_count = np.sum(jnt_visible, axis=1)
#
#         less_than_threshold = np.multiply((scaled_uv_err <= threshold), jnt_visible)
#
#         PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)
#
#         # save
#         rng = np.arange(0, 0.5 + 0.01, 0.01)
#         pckAll = np.zeros((len(rng), 23))
#
#         for r in range(len(rng)):
#             threshold = rng[r]
#             less_than_threshold = np.multiply(scaled_uv_err <= threshold, jnt_visible)
#             pckAll[r, :] = np.divide(100. * np.sum(less_than_threshold, axis=1),
#                                      jnt_count)
#
#         jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)
#         name_value = [
#             ('Head', PCKh[head]),
#             ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
#             ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
#             ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
#             ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
#             ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
#             ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
#             ('Mouth', 0.25 * (PCKh[tmouth] + PCKh[lmouth] + PCKh[rmouth] + PCKh[bmouth])),
#             ('Tail', (PCKh[mtail] + PCKh[btail] + PCKh[ttail]) / 3),
#             ('Mean', np.sum(PCKh * jnt_ratio)),
#             # ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
#         ]
#         name_value = OrderedDict(name_value)
#
#         return name_value, name_value['Mean']
#
#     # def ReturnWeight(cfg):
#     #     weight = torch.nn.Parameter(torch.ones((cfg.MODEL.NUM_JOINTS, 1), dtype=torch.float32).cuda())
#     #
#     #     return weight

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import logging
import os
import json_tricks as json
from collections import OrderedDict
import json
import numpy as np
import torch
from scipy.io import loadmat, savemat

from dataset.JointsDataset import JointsDataset

logger = logging.getLogger(__name__)


class AnimalKingdomDataset(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)
        '''
        0: Head_Mid_Top
        1: Eye_Left
        2: Eye_Right
        3: Mouth_Front_Top
        4: Mouth_Back_Left
        5: Mouth_Back_Right
        6: Mouth_Front_Bottom
        7: Shoulder_Left
        8: Shoulder_Right
        9: Elbow_Left
        10: Elbow_Right
        11: Wrist_Left
        12: Wrist_Right
        13: Torso_Mid_Back
        14: Hip_Left
        15: Hip_Right
        16: Knee_Left
        17: Knee_Right
        18: Ankle_Left
        19: Ankle_Right
        20: Tail_Top_Back
        21: Tail_Mid_Back
        22: Tail_End_Back
        '''
        ### Changes below ###
        self.num_joints = 23
        self.flip_pairs = [[1, 2], [4, 5], [7, 8], [9, 10], [11, 12], [14, 15], [16, 17], [18, 19]]
        self.parent_ids = [3, 3, 3, 6, 6, 6, 13, 13, 13, 7, 8, 9, 10, 13, 13, 13, 14, 15, 16, 17, 12, 20, 21]

        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
        self.lower_body_ids = (14, 15, 16, 17, 18, 19, 20, 21, 22)

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # Create train/val split
        file_name = os.path.join(
            self.root, 'pose_estimation/annotation/ak_P1', self.image_set + '.json'
        )
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        bbox_id = 0
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float)
            s = np.array([a['scale'], a['scale']], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                # c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            #             # we should first convert to 0-based index
            #             c = c - 1

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            if self.image_set != 'test':
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2]
                joints_vis = np.array(a['joints_vis'])
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            # image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
            image_dir = 'images.zip@' if self.data_format == 'zip' else 'pose_estimation/dataset/dataset'
            gt_db.append(
                {
                    'image': os.path.join(self.root, image_dir, image_name),
                    'center': c,
                    'scale': s,
                    'joints_3d': joints_3d,
                    'joints_3d_vis': joints_3d_vis,
                    'filename': '',
                    'imgnum': 0,
                    'bbox_score': 1,
                    'bbox_id': bbox_id
                }
            )
            bbox_id = bbox_id + 1

        return gt_db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # Convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.mat')
            savemat(pred_file, mdict={'preds': preds})

        #         if 'test' in cfg.DATASET.TEST_SET:
        #             return {'Null': 0.0}, 0.0

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

    # def ReturnWeight(cfg):
    #     weight = torch.nn.Parameter(torch.ones((cfg.MODEL.NUM_JOINTS, 1), dtype=torch.float32).cuda())
    #
    #     return weight

# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
# import math
# import logging
# import os
#
# from pycocotools.coco import COCO
# from pycocotools.cocoeval import COCOeval
# import json_tricks as json
# from collections import OrderedDict
# import json
# import numpy as np
# import torch
# from scipy.io import loadmat, savemat
#
# from dataset.JointsDataset import JointsDataset
#
# logger = logging.getLogger(__name__)
#
#
# class AnimalKingdomDataset(JointsDataset):
#     def __init__(self, cfg, root, image_set, is_train, transform=None):
#         super().__init__(cfg, root, image_set, is_train, transform)
#         '''
#         0: Head_Mid_Top
#         1: Eye_Left
#         2: Eye_Right
#         3: Mouth_Front_Top
#         4: Mouth_Back_Left
#         5: Mouth_Back_Right
#         6: Mouth_Front_Bottom
#         7: Shoulder_Left
#         8: Shoulder_Right
#         9: Elbow_Left
#         10: Elbow_Right
#         11: Wrist_Left
#         12: Wrist_Right
#         13: Torso_Mid_Back
#         14: Hip_Left
#         15: Hip_Right
#         16: Knee_Left
#         17: Knee_Right
#         18: Ankle_Left
#         19: Ankle_Right
#         20: Tail_Top_Back
#         21: Tail_Mid_Back
#         22: Tail_End_Back
#         '''
#         ### Changes below ###
#         self.num_joints = 23
#         self.flip_pairs = [[1, 2], [4, 5], [7, 8], [9, 10], [11, 12], [14, 15], [16, 17], [18, 19]]
#         self.parent_ids = [3, 3, 3, 6, 6, 6, 13, 13, 13, 7, 8, 9, 10, 13, 13, 13, 14, 15, 16, 17, 12, 20, 21]
#         self.use_gt_bbox = cfg.TEST.COCO_BBOX_FILE
#         self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13)
#         self.lower_body_ids = (14, 15, 16, 17, 18, 19, 20, 21, 22)
#
#         # logger.info('=> annotation_path:{}'.format(self._get_ann_file_keypoint()))
#         self.coco = COCO(self._get_ann_file_keypoint())
#         self.image_set_index = self.coco.getImgIds()
#         # deal with class names
#         cats = [cat['name']
#                 for cat in self.coco.loadCats(self.coco.getCatIds())]
#         self.classes = ['__background__'] + cats
#         logger.info('=> classes: {}, and the number of classes is {}'.format(self.classes, len(self.classes)))
#         self.num_classes = len(self.classes)
#         self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
#         self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
#         self._coco_ind_to_class_ind = dict(
#             [
#                 (self._class_to_coco_ind[cls], self._class_to_ind[cls])
#                 for cls in self.classes[1:]
#             ]
#         )
#
#         self.db = self._get_db()
#
#         if is_train and cfg.DATASET.SELECT_DATA:
#             self.db = self.select_data(self.db)
#
#         logger.info('=> load {} samples'.format(len(self.db)))
#
#     def _get_ann_file_keypoint(self):
#         """ self.root / annotations / ap10k_train_split1.json """
#         return os.path.join(
#             'data/ak/pose_estimation/annotation/ak_P1/' + self.image_set + '.json'
#         )
#
#     ######## origin ##############
#     # def _get_db(self):
#     #     # Create train/val split
#     #     file_name = os.path.join(
#     #         self.root, 'annot', self.image_set + '.json'
#     #     )
#     #     with open(file_name) as anno_file:
#     #         anno = json.load(anno_file)
#     #
#     #     gt_db = []
#     #     bbox_id = 0
#     #     for a in anno:
#     #         image_name = a['image']
#     #
#     #         c = np.array(a['center'], dtype=np.float)
#     #         s = np.array([a['scale'], a['scale']], dtype=np.float)
#     #
#     #         # Adjust center/scale slightly to avoid cropping limbs
#     #         if c[0] != -1:
#     #             # c[1] = c[1] + 15 * s[1]
#     #             s = s * 1.25
#     #
#     #         # MPII uses matlab format, index is based 1,
#     #         #             # we should first convert to 0-based index
#     #         #             c = c - 1
#     #
#     #         joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
#     #         joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
#     #         # if self.image_set != 'test':
#     #         joints = np.array(a['joints'])
#     #         joints[:, 0:2] = joints[:, 0:2]
#     #         joints_vis = np.array(a['joints_vis'])
#     #         assert len(joints) == self.num_joints, \
#     #             'joint num diff: {} vs {}'.format(len(joints),
#     #                                               self.num_joints)
#     #
#     #         joints_3d[:, 0:2] = joints[:, 0:2]
#     #         joints_3d_vis[:, 0] = joints_vis[:]
#     #         joints_3d_vis[:, 1] = joints_vis[:]
#     #
#     #         image_dir = 'images.zip@' if self.data_format == 'zip' else 'images'
#     #         gt_db.append(
#     #             {
#     #                 'image': os.path.join(self.root, image_dir, image_name),
#     #                 'center': c,
#     #                 'scale': s,
#     #                 'joints_3d': joints_3d,
#     #                 'joints_3d_vis': joints_3d_vis,
#     #                 'filename': '',
#     #                 'imgnum': 0,
#     #                 'bbox_score': 1,
#     #                 'bbox_id': bbox_id
#     #             }
#     #         )
#     #         bbox_id = bbox_id + 1
#     #
#     #     return gt_db
#     ###############################
#
#     def image_path_from_index(self, index):
#         """ example: images / 000000119993.jpg """
#         file_name = '%012d.jpg' % index
#
#         # prefix = self.image_set
#
#         # data_name = prefix + '.zip@' if self.data_format == 'zip' else prefix
#
#         image_path = os.path.join(
#             self.root, 'pose_estimation/dataset/dataset', file_name)
#
#         return image_path
#
#     def _box2cs(self, box):
#         x, y, w, h = box[:4]
#         return self._xywh2cs(x, y, w, h)
#
#     def _xywh2cs(self, x, y, w, h):
#         center = np.zeros((2), dtype=np.float32)
#         center[0] = x + w * 0.5
#         center[1] = y + h * 0.5
#
#         if w > self.aspect_ratio * h:
#             h = w * 1.0 / self.aspect_ratio
#         elif w < self.aspect_ratio * h:
#             w = h * self.aspect_ratio
#         scale = np.array(
#             [w * 1.0 / self.pixel_std, h * 1.0 / self.pixel_std],
#             dtype=np.float32)
#         if center[0] != -1:
#             scale = scale * 1.25
#
#         return center, scale
#
#     def _get_db(self):
#         gt_db = self._load_coco_keypoint_annotations()
#         return gt_db
#
#     def _load_coco_keypoint_annotations(self):
#         gt_db = []
#         for index in self.image_set_index:
#             gt_db.extend(self._load_coco_keypoint_annotations_kernel(index))
#         return gt_db
#
#     def _load_coco_keypoint_annotations_kernel(self, index):
#         img_ann = self.coco.loadImgs(index)[0]
#         width = img_ann['width']
#         height = img_ann['height']
#         num_joints = self.num_joints
#
#         annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
#         objs = self.coco.loadAnns(annIds)
#
#         # sanitize bboxes
#         valid_objs = []
#         for obj in objs:
#             if 'bbox' not in obj:
#                 continue
#             x, y, w, h = obj['bbox']
#             x1 = np.max((0, x))
#             y1 = np.max((0, y))
#             x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
#             y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
#             if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
#                 obj['clean_bbox'] = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
#                 valid_objs.append(obj)
#         objs = valid_objs
#
#         bbox_id = 0
#         rec = []
#         for obj in objs:
#             if 'keypoints' not in obj:
#                 continue
#             if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
#                 continue
#             cls = self._coco_ind_to_class_ind[obj['category_id']]
#             if cls == 0:
#                 continue
#
#             # ignore objs without keypoints annotation
#             if max(obj['keypoints']) == 0:
#                 continue
#
#             joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
#             joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
#             for ipt in range(self.num_joints):
#                 joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
#                 joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
#                 joints_3d[ipt, 2] = 0
#                 t_vis = obj['keypoints'][ipt * 3 + 2]
#                 if t_vis > 1:
#                     t_vis = 1
#                 joints_3d_vis[ipt, 0] = t_vis
#                 joints_3d_vis[ipt, 1] = t_vis
#                 joints_3d_vis[ipt, 2] = 0
#
#             center, scale = self._box2cs(obj['clean_bbox'][:4])
#
#             # image_file = os.path.join(self.root, 'images', self.id2name[index])
#             image_dir = 'images.zip@' if self.data_format == 'zip' else 'pose_estimation/dataset/dataset'
#             image_name = img_ann['file_name']
#             rec.append({
#                 'image': os.path.join(self.root + '/' + image_dir + '/' + image_name),
#                 # os.path.join(self.root, image_dir, self.id2name[index]),
#                 'bboxes': obj['clean_bbox'][:4],
#                 'center': center,
#                 'scale': scale,
#                 'joints_3d': joints_3d,
#                 'joints_3d_vis': joints_3d_vis,
#                 'filename': '',
#                 'imgnum': 0,
#                 'bbox_score': 1,
#                 'bbox_id': bbox_id
#             })
#             bbox_id = bbox_id + 1
#         return rec
#
#     def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
#         # Convert 0-based index to 1-based index
#         preds = preds[:, :, 0:2] + 1.0
#
#         if output_dir:
#             pred_file = os.path.join(output_dir, 'pred.mat')
#             savemat(pred_file, mdict={'preds': preds})
#
#         #         if 'test' in cfg.DATASET.TEST_SET:
#         #             return {'Null': 0.0}, 0.0
#
#         SC_BIAS = 1
#         threshold = 0.05
#
#         gt_file = os.path.join(cfg.DATASET.ROOT,
#                                'pose_estimation/annotation/ak_P1',
#                                '{}.json'  # gt_{}.json'
#                                .format(cfg.DATASET.TEST_SET)
#                                )
#
#         with open(gt_file) as f:
#             gt_dict = json.load(f)
#
#         ### Changes below ###
#
#         # dataset_joints = gt_dict['dataset_joints']
#         # jnt_visible = [v for k, v in gt_dict['joints_vis'].items()]
#         # pos_gt_src = [v for k, v in gt_dict['joints'].items()]
#         # scale = [v for k, v in gt_dict['scale'].items()]
#
#         dataset_joints = [
#             [
#                 "Head_Mid_Top"
#             ],
#             [
#                 "Eye_Left"
#             ],
#             [
#                 "Eye_Right"
#             ],
#             [
#                 "Mouth_Front_Top"
#             ],
#             [
#                 "Mouth_Back_Left"
#             ],
#             [
#                 "Mouth_Back_Right"
#             ],
#             [
#                 "Mouth_Front_Bottom"
#             ],
#             [
#                 "Shoulder_Left"
#             ],
#             [
#                 "Shoulder_Right"
#             ],
#             [
#                 "Elbow_Left"
#             ],
#             [
#                 "Elbow_Right"
#             ],
#             [
#                 "Wrist_Left"
#             ],
#             [
#                 "Wrist_Right"
#             ],
#             [
#                 "Torso_Mid_Back"
#             ],
#             [
#                 "Hip_Left"
#             ],
#             [
#                 "Hip_Right"
#             ],
#             [
#                 "Knee_Left"
#             ],
#             [
#                 "Knee_Right"
#             ],
#             [
#                 "Ankle_Left"
#             ],
#             [
#                 "Ankle_Right"
#             ],
#             [
#                 "Tail_Top_Back"
#             ],
#             [
#                 "Tail_Mid_Back"
#             ],
#             [
#                 "Tail_End_Back"
#             ]
#         ]
#
#         jnt_visible = [x['joints_vis'] for x in gt_dict]
#         pos_gt_src = [x['joints'] for x in gt_dict]
#         scale = [x['scale'] for x in gt_dict]
#
#         scale = np.array(scale)
#         scale = scale * 200 * math.sqrt(2)
#         threshold_bbox = []
#         for item in self.db:
#             bbox = np.array(item['bboxes'])
#             bbox_thr = np.max(bbox[2:])
#             threshold_bbox.append(np.array(bbox_thr))
#
#         threshold_bbox = np.array(threshold_bbox)
#         # bbox = [x['bboxes'] for x in self.db]
#         # #
#         # bbox = np.array(bbox)
#         # bbox_thr = np.max(bbox[2:])
#         # scale = scale * 200 * math.sqrt(2)
#
#         jnt_visible = np.transpose(jnt_visible, [1, 0])
#         pos_pred_src = np.transpose(preds, [1, 2, 0])
#         pos_gt_src = np.transpose(pos_gt_src, [1, 2, 0])
#         dataset_joints = np.array(dataset_joints)
#         head = np.where(dataset_joints == 'Head_Mid_Top')[0][0]
#         lsho = np.where(dataset_joints == 'Shoulder_Left')[0][0]
#         lelb = np.where(dataset_joints == 'Elbow_Left')[0][0]
#         lwri = np.where(dataset_joints == 'Wrist_Left')[0][0]
#         lhip = np.where(dataset_joints == 'Hip_Left')[0][0]
#         lkne = np.where(dataset_joints == 'Knee_Left')[0][0]
#         lank = np.where(dataset_joints == 'Ankle_Left')[0][0]
#
#         rsho = np.where(dataset_joints == 'Shoulder_Right')[0][0]
#         relb = np.where(dataset_joints == 'Elbow_Right')[0][0]
#         rwri = np.where(dataset_joints == 'Wrist_Right')[0][0]
#         rhip = np.where(dataset_joints == 'Hip_Right')[0][0]
#         rkne = np.where(dataset_joints == 'Knee_Right')[0][0]
#         rank = np.where(dataset_joints == 'Ankle_Right')[0][0]
#
#         tmouth = np.where(dataset_joints == 'Mouth_Front_Top')[0][0]
#         lmouth = np.where(dataset_joints == 'Mouth_Back_Left')[0][0]
#         rmouth = np.where(dataset_joints == 'Mouth_Back_Right')[0][0]
#         bmouth = np.where(dataset_joints == 'Mouth_Front_Bottom')[0][0]
#         ttail = np.where(dataset_joints == 'Tail_Top_Back')[0][0]
#         mtail = np.where(dataset_joints == 'Tail_Mid_Back')[0][0]
#         btail = np.where(dataset_joints == 'Tail_End_Back')[0][0]
#
#         uv_error = pos_pred_src - pos_gt_src
#
#         uv_err = np.linalg.norm(uv_error, axis=1)
#
#         #         headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
#         #         headsizes = np.linalg.norm(headsizes, axis=0)
#         # scale *= SC_BIAS
#         # headsizes = scale
#         threshold_bbox *= SC_BIAS
#         length = threshold_bbox
#
#         scale = np.multiply(length, np.ones((len(uv_err), 1)))
#         scaled_uv_err = np.divide(uv_err, scale)
#         scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
#         jnt_count = np.sum(jnt_visible, axis=1)
#
#         less_than_threshold = np.multiply((scaled_uv_err <= threshold),
#                                           jnt_visible)
#         PCKh = np.divide(100. * np.sum(less_than_threshold, axis=1), jnt_count)
#         # save
#         rng = np.arange(0, 0.5 + 0.01, 0.01)
#         pckAll = np.zeros((len(rng), 23))
#
#         for r in range(len(rng)):
#             threshold = rng[r]
#             less_than_threshold = np.multiply(scaled_uv_err <= threshold,
#                                               jnt_visible)
#             pckAll[r, :] = np.divide(100. * np.sum(less_than_threshold, axis=1),
#                                      jnt_count)
#
#         #         PCKh = np.ma.array(PCKh, mask=False)
#         #         PCKh.mask[21:22] = True
#
#         #         jnt_count = np.ma.array(jnt_count, mask=False)
#         #         jnt_count.mask[21:22] = True
#
#         jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)
#         name_value = [
#             ('Head', PCKh[head]),
#             ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
#             ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
#             ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
#             ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
#             ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
#             ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
#             ('Mouth', 0.25 * (PCKh[tmouth] + PCKh[lmouth] + PCKh[rmouth] + PCKh[bmouth])),
#             ('Tail', (PCKh[ttail] + PCKh[mtail] + PCKh[btail]) / 3),
#             ('Mean', np.sum(PCKh * jnt_ratio))
#             #             ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
#         ]
#         name_value = OrderedDict(name_value)
#         return name_value, name_value['Mean']
#
#     def ReturnWeight(cfg):
#         weight = torch.nn.Parameter(torch.ones((cfg.MODEL.NUM_JOINTS, 1), dtype=torch.float32).cuda())
#
#         return weight
