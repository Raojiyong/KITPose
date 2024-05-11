# Written by Jiyong Rao
# ----------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def keypoint_info(ds_name):
    if ds_name == 'ap10k':
        keypoint_names = ['leye', 'reye', 'nose', 'neck', 'tail', 'lsho', 'lelb', 'lfpaw', 'rsho',
                          'relb', 'rfpaw', 'lhip', 'lknee', 'lbpaw', 'rhip', 'rknee', 'rbpaw']

        keypoint_dependence = {'leye': ['reye', 'nose'], 'reye': ['leye', 'nose'],
                               'nose': ['leye', 'reye', 'neck'],
                               'neck': ['nose', 'lsho', 'rsho'], 'lsho': ['neck', 'lelb'],
                               'lelb': ['lsho', 'lfpaw'], 'lfpaw': ['lelb'], 'rsho': ['neck', 'relb'],
                               'relb': ['rsho', 'rfpaw'], 'rfpaw': ['relb'], 'tail': ['lhip', 'rhip'],
                               'lhip': ['tail', 'lknee'], 'lknee': ['lhip', 'lbpaw'], 'lbpaw': ['lknee'],
                               'rhip': ['tail', 'rknee'], 'rknee': ['rhip', 'rbpaw'], 'rbpaw': ['rknee']
                               }
        kpt_dict = {
            'leye': 0, 'reye': 1, 'nose': 2, 'neck': 3, 'tail': 4, 'lsho': 5, 'lelb': 6, 'lfpaw': 7, 'rsho': 8,
            'relb': 9, 'rfpaw': 10, 'lhip': 11, 'lknee': 12, 'lbpaw': 13, 'rhip': 14, 'rknee': 15, 'rbpaw': 16
        }
        # keypoint_dependence = {}
        # for kpt in keypoint_names:
        #     keypoint_dependence[kpt] = [kpt_cond for kpt_cond in keypoint_names if kpt_cond != kpt]

    elif ds_name == 'atrw':
        keypoint_names = ["lear", "rear", "nose", "rsho", "rfpaw", "lsho",
                          "lfpaw", "rhip", "rknee", "rbpaw", "lhip", "lknee",
                          "lbpaw", "tail", "center"]

        keypoint_dependence = {'lear': ['nose', 'rear'], 'rear': ['nose', 'lear'],
                               'nose': ['lear', 'rear'],
                               'lsho': ['lfpaw', 'center'], 'rsho': ['rfpaw', 'center'],
                               'rfpaw': ['rsho'], 'lfpaw': ['lsho'],
                               'tail': ['center', 'rhip', 'lhip'], 'rbpaw': ['rknee'],
                               'rknee': ['rhip', 'rbpaw'], 'rhip': ['rknee', 'tail'],
                               'lbpaw': ['lknee'], 'lknee': ['lhip', 'lbpaw'],
                               'lhip': ['tail', 'lknee'], 'center': ['lsho', 'rsho', 'tail']
                               }

        kpt_dict = {
            'lear': 0, 'rear': 1, 'nose': 2, 'rsho': 3, 'rfpaw': 4, 'lsho': 5,
            'lfpaw': 6, 'rhip': 7, 'rknee': 8, 'rbpaw': 9, 'lhip': 10, 'lknee': 11,
            'lbpaw': 12, 'tail': 13, 'center': 14
        }
        # keypoint_dependence = {}
        # for kpt in keypoint_names:
        #     keypoint_dependence[kpt] = [kpt_cond for kpt_cond in keypoint_names if kpt_cond != kpt]

    else:
        raise ValueError('the dataset {} is not exist!'.format(ds_name))
    return keypoint_names, keypoint_dependence, kpt_dict
