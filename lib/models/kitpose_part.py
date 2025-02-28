from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers.weight_init import trunc_normal_
import numpy as np
import math
from .fast_keans import batch_fast_kmedoids_with_split
from .pose_hrnet import PoseHighResolutionNet, Bottleneck, BasicBlock

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, inner_dim, heads=8, dropout=0., num_keypoints=None, scale_with_head=False):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5 if scale_with_head else dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.num_keypoints = num_keypoints

    # @get_local('attn')
    def forward(self, x, mask=None, pos_embed=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class PoseEncoderLayer(nn.Module):
    def __init__(self, dim, inner_dim, heads, dropout, mlp_dim, in_channels=None, all_attn=False,
                 scale_with_head=False):
        super().__init__()
        self.attn = Attention(dim, inner_dim, heads=heads, dropout=dropout, num_keypoints=in_channels,
                              scale_with_head=scale_with_head)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None, pos=None):
        residual = x
        x = self.norm1(x)
        x = self.attn(x, mask, pos)
        x += residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        out = x + residual
        return out


class Transformer(nn.Module):
    def __init__(self, dim, inner_dim, depth, heads, mlp_dim, dropout, num_keypoints=None, num_bp=None, all_attn=False,
                 scale_with_head=False):
        super().__init__()
        self.all_attn = all_attn
        self.num_keypoints = num_keypoints
        self.num_bp = num_bp
        self.layers = nn.ModuleList([
            PoseEncoderLayer(dim, inner_dim, heads, dropout, mlp_dim, num_keypoints + num_bp, all_attn, scale_with_head)
            for _ in range(depth)
        ])

    def forward(self, x, mask=None, pos=None):
        # normal cnnect
        for idx, layer in enumerate(self.layers):
            if idx > 0 and self.all_attn:
                x[:, self.num_bp:] += pos
            x = layer(x, mask=mask, pos=pos)
        return x


class PromptLearner(nn.Module):
    def __init__(self, bp_num, dim, hidden_dim, in_planes):
        super().__init__()
        self.meta_net = nn.Sequential(
            nn.Conv2d(in_planes, bp_num, 3, 1, 1),
            nn.BatchNorm2d(bp_num),
            nn.ReLU(),
            nn.Conv2d(bp_num, bp_num, 3, 1, 1),
            nn.BatchNorm2d(bp_num),
            nn.ReLU(),
        )

        self.prompt_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.LayerNorm(dim // 2),
            nn.Linear(dim // 2, hidden_dim),
        )

    def forward(self, x, bp_prompt):
        p = x.shape[-2:]
        bias = self.meta_net(x)
        bias = rearrange(bias, 'b c p1 p2 -> b c (p1 p2)', p1=p[0], p2=p[1])
        bp_prompt = bp_prompt + bias
        bp_prompt = self.prompt_head(bp_prompt)
        return bp_prompt


class KITPose_base(nn.Module):
    def __init__(self, *, feature_size, kpt_size, num_keypoints, num_bp, dim, inner_dim, depth, heads, mlp_dim,
                 apply_init=False, apply_multi=True, hidden_heatmap_dim=64 * 32, heatmap_dim=64 * 64,
                 heatmap_size=[64, 64], channels=32, dropout=0., emb_dropout=0., pos_embedding_type="learnable"):
        super().__init__()
        assert isinstance(feature_size, list) and isinstance(kpt_size,
                                                             list), 'image_size and patch_size should be list'
        assert feature_size[0] % kpt_size[0] == 0 and feature_size[1] % kpt_size[
            1] == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (feature_size[0] // (kpt_size[0])) * (feature_size[1] // (kpt_size[1])) * num_keypoints
        kpt_dim = kpt_size[0] * kpt_size[1]
        assert pos_embedding_type in ['none', 'sine', 'learnable', 'sine-full']

        self.inplanes = channels
        self.patch_size = kpt_size
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.num_bp = num_bp
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = (self.pos_embedding_type == "sine-full")

        self.bp_embeddings = nn.Parameter(torch.zeros(1, self.num_bp, dim))
        h, w = feature_size[0] // (self.patch_size[0]), feature_size[1] // (self.patch_size[1])
        self._make_position_embedding(w, h, dim, inner_dim, pos_embedding_type)

        self.kpt_to_embedding = nn.Linear(kpt_dim, dim)
        self.bp_to_embeddings = nn.Linear(kpt_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        # transformer
        self.transformer = Transformer(dim, inner_dim, depth, heads, mlp_dim, dropout, num_keypoints=num_keypoints,
                                       num_bp=num_bp, all_attn=self.all_attn, scale_with_head=False)

        self.to_keypoint_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_heatmap_dim),
            nn.LayerNorm(hidden_heatmap_dim),
            nn.Linear(hidden_heatmap_dim, heatmap_dim)
        ) if (dim <= hidden_heatmap_dim * 0.5 and apply_multi) else nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, heatmap_dim)
        )
        trunc_normal_(self.bp_embeddings, std=.02)
        self.prompt_learner = PromptLearner(num_bp, heatmap_dim, dim, self.inplanes)

        if apply_init:
            self.apply(self._init_weights)

    def _make_position_embedding(self, w, h, d_model, d_inner=256, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_keypoints + self.num_bp, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding_1d(self.num_keypoints, d_model),
                    requires_grad=False)

                print("==> Add Sine PositionEmbedding~")

    def _make_sine_position_embedding_2d(self, d_model, temperature=10000,
                                         scale=2 * math.pi):
        h, w = self.pe_h, self.pe_w
        area = torch.ones(1, h, w)  # [b, h, w]
        y_embed = area.cumsum(1, dtype=torch.float32)  # [1, h, w]
        x_embed = area.cumsum(2, dtype=torch.float32)  # [1, h, w]

        one_direction_feats = d_model // 2

        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(one_direction_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / one_direction_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = pos.flatten(2).permute(0, 2, 1)
        return pos

    def _make_sine_position_embedding_1d(self, channels, d_model, temperature=10000, scale=2 * math.pi):
        embed = torch.ones((1, channels))
        embed = embed.cumsum(1, dtype=torch.float32)
        eps = 1e-6
        embed = embed / (embed[:, -1:] + eps) * scale
        dim_t = torch.arange(d_model, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / d_model)

        pos = embed[:, :, None] / dim_t
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)  # 1, num_joints, embed_dim
        return pos

    def _init_weights(self, m):
        # print("Initialization...")
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, all_feature, kpt_feature, mask=None):
        p = self.patch_size
        # transformer
        kpt_feature = rearrange(kpt_feature, 'b c p1 p2 -> b c (p1 p2)', p1=p[0], p2=p[1])
        # all_feature = rearrange(all_feature, 'b c p1 p2 -> b c (p1 p2)', p1=p[0], p2=p[1])
        x = self.kpt_to_embedding(kpt_feature)
        # bp_embeddings = self.bp_to_embeddings(all_feature)

        b, n, _ = x.shape

        cluster_num = self.num_bp

        assign, mediods_ids = batch_fast_kmedoids_with_split(kpt_feature, cluster_num,
                                                             distance='cosine', threshold=1e-6,
                                                             iter_limit=40,
                                                             id_sort=True,
                                                             norm_p=2.0,
                                                             split_size=8,
                                                             pre_norm=True)

        bp_emb_list = []
        for i in range(cluster_num):
            # [B, cluster, 1]
            bp_mask = (assign == i).unsqueeze(-1)
            # [B, 1, width]
            x_tmp_tmp = torch.sum(kpt_feature * bp_mask, dim=1, keepdim=True) / torch.sum(
                bp_mask.float(), dim=1, keepdim=True)
            bp_emb_list.append(x_tmp_tmp)
        # [B x T_new, cluster, width]
        bp_emb_tmp = torch.cat(bp_emb_list, dim=1)

        # bp_embeddings = self.bp_to_embeddings(bp_emb_tmp)
        bp_embeddings = self.prompt_learner(all_feature, bp_emb_tmp)
        # bp_embeddings = repeat(self.bp_embeddings, '() n d -> b n d', b=b)
        if self.pos_embedding_type in ["sine", "sine-full"]:
            x += self.pos_embedding[:, :n]
            x = torch.cat((bp_embeddings, x), dim=1)
        else:
            x = torch.cat((bp_embeddings, x), dim=1)
            x += self.pos_embedding[:, :(n + self.num_keypoints)]
            # x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x, mask, self.pos_embedding)
        x = self.mlp_head(x)
        x = rearrange(x, 'b c (p1 p2) -> b c p1 p2', p1=self.heatmap_size[0], p2=self.heatmap_size[1])
        bp_embeddings, x = \
            self.to_keypoint_token(x[:, :self.num_bp]), self.to_keypoint_token(x[:, self.num_bp:])
        return bp_embeddings, x


class KITPose(nn.Module):
    def __init__(self, cfg, **kwargs):
        extra = cfg.MODEL.EXTRA
        super(KITPose, self).__init__()

        self.pre_feature = PoseHighResolutionNet(cfg, **kwargs)
        # self.pre_feature = HRNET_base(cfg, map=True, **kwargs)
        self.channelformer = KITPose_base(
            feature_size=[cfg.MODEL.IMAGE_SIZE[1] // 4, cfg.MODEL.IMAGE_SIZE[0] // 4],
            kpt_size=[cfg.MODEL.PATCH_SIZE[1], cfg.MODEL.PATCH_SIZE[0]],
            num_keypoints=cfg.MODEL.NUM_JOINTS, num_bp=cfg.MODEL.NUM_BP,
            dim=cfg.MODEL.DIM, inner_dim=cfg.MODEL.INNER_DIM,
            channels=extra.STAGE2.NUM_CHANNELS[0],
            depth=cfg.MODEL.TRANSFORMER_DEPTH, heads=cfg.MODEL.TRANSFORMER_HEADS,
            mlp_dim=cfg.MODEL.DIM * cfg.MODEL.TRANSFORMER_MLP_RATIO,
            apply_init=cfg.MODEL.INIT,
            hidden_heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1] * cfg.MODEL.HEATMAP_SIZE[0] // 8,
            heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1] * cfg.MODEL.HEATMAP_SIZE[0],
            heatmap_size=[cfg.MODEL.HEATMAP_SIZE[1], cfg.MODEL.HEATMAP_SIZE[0]],
            pos_embedding_type=cfg.MODEL.POS_EMBEDDING_TYPE,
        )

    def forward(self, x):
        all_feats, kpt_feats = self.pre_feature(x)
        bp, out = self.channelformer(all_feats, kpt_feats)
        return kpt_feats, bp, out

    def init_weights(self, pretrained=''):
        self.pre_feature.init_weights(pretrained)


def get_pose_net(cfg, is_train, **kwargs):
    model = KITPose(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model
