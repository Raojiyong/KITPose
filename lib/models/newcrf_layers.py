from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np

from einops import rearrange, repeat
from timm.models.layers.weight_init import trunc_normal_, DropPath, to_2tuple
import math
# from .kitpose_base import KITPose_base
from .hrnet import HRNET_base

# from .hr_base import HRNET_base

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class Mlp(nn.Module):
    """ Multilayer perceptron"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)


def get_pose_net(cfg, is_train, **kwargs):
    model = KITPose(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model
