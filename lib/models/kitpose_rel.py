from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from timm.models.layers.weight_init import trunc_normal_
import math
from .pose_hrnet import PoseHighResolutionNet
#
# from .hr_base import HRNET4_base

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
    def __init__(self, dim, heads=8, dropout=0., num_keypoints=None, scale_with_head=False):
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
    def forward(self, x, mask=None):
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
        return out, attn


class RelMultiHeadAttn(nn.Module):
    def __init__(self, dim, heads, dropout=0., scale_with_head=False):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (dim // heads) ** -0.5 if scale_with_head else dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim, bias=False),
            nn.Dropout(dropout)
        )

    def _rel_shift(self, x, zero_triu=False):
        # x (b,h,i,j)
        zero_pad = torch.zeros((*x.size()[:2], x.size(2), 1),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)
        x_padded = x_padded.view(x.size(3) + 1, x.size(2), *x.size()[:2])

        x = x_padded[1:].view_as(x)

        if zero_triu:
            ones = torch.ones((x.size(0), x.size(1)))
            x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]
        return x

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None):
        raise NotImplementedError


class RelativeLearnableAttention(RelMultiHeadAttn):
    def __init__(self, dim, heads, dropout=0., scale_with_head=False):
        super().__init__(dim, heads)
        self.to_r = nn.Linear(dim, dim, bias=False)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None):
        b, n, _, h = *w.shape, self.heads

        qkv = self.to_qkv(w).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        r = self.to_r(r)
        r = r.view(h, n, self.dim)

        # compute attention score
        # q,k,v (b,h,n,d)
        rw_q = q + r_w_bias
        AC = torch.einsum('bhid,bhjd->bhij', rw_q, k)

        rr_q = q + r_r_bias
        BD = torch.einsum('bhid,hjd->bhij', rr_q, r)
        # BD = self._rel_shift(BD)
        attn_prob = AC + BD
        attn_prob = attn_prob * self.scale

        if attn_mask is not None:
            mask_value = -torch.finfo(attn_prob.dtype).max
            mask = F.pad(attn_mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == attn_prob.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            attn_prob.masked_fill_(~mask, mask_value)
            del mask

        attn = attn_prob.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, attn


class PoseEncoderLayer(nn.Module):
    def __init__(self, dim, heads, dropout, mlp_dim, num_keypoints=None, all_attn=False, scale_with_head=False,
                 attn_type=0):
        super().__init__()
        if attn_type == 0:
            self.attn = Attention(dim, heads=heads, dropout=dropout, num_keypoints=num_keypoints,
                                  scale_with_head=scale_with_head)
        elif attn_type == 1:
            self.attn = RelativeLearnableAttention(dim, heads, dropout=0., scale_with_head=scale_with_head)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, pos, r_w_bias, r_r_bias, mask=None):
        residual = x
        x = self.norm1(x)
        x, attn = self.attn(x, pos, r_w_bias, r_r_bias, mask)
        x += residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        out = x + residual
        return out, attn


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, num_keypoints=None, all_attn=False, scale_with_head=False,
                 attn_type=0):
        super().__init__()
        self.layers = nn.ModuleList([
            PoseEncoderLayer(dim, heads, dropout, mlp_dim, num_keypoints, all_attn, scale_with_head,
                             attn_type=attn_type)
            for _ in range(depth)
        ])

    def forward(self, x, pos, r_w_bias, r_r_bias, mask=None):
        features = [x]
        for layer in self.layers:
            feature = torch.stack(features).sum(dim=0)
            new_feature, attn_weights = layer(feature, pos, r_w_bias, r_r_bias, mask=mask)
            features.append(new_feature)
        out = torch.stack(features).sum(dim=0)
        return out


class KITPose_base(nn.Module):
    def __init__(self, *, feature_size, patch_size, num_keypoints, dim, depth, heads, mlp_dim, attn_type=0,
                 apply_init=False, apply_multi=True, hidden_heatmap_dim=64 * 32, heatmap_dim=64 * 64,
                 heatmap_size=[64, 64], dropout=0., emb_dropout=0., pos_embedding_type="learnable"):
        super().__init__()
        assert isinstance(feature_size, list) and isinstance(patch_size,
                                                             list), 'image_size and patch_size should be list'
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[
            1] == 0, 'Image dimensions must be divisible by the patch size.'
        patch_dim = patch_size[0] * patch_size[1]
        assert pos_embedding_type in ['none', 'sine', 'learnable', 'sine-full', 'relative']

        self.inplanes = 64
        self.patch_size = patch_size
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.pos_embedding_type = pos_embedding_type
        self.heads = heads
        self.all_attn = (self.pos_embedding_type == "sine-full")

        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        h, w = feature_size[0] // (self.patch_size[0]), feature_size[1] // (self.patch_size[1])
        self._make_position_embedding(w, h, dim, pos_embedding_type)

        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        # transformer
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, num_keypoints=num_keypoints,
                                       all_attn=self.all_attn, scale_with_head=False, attn_type=attn_type)

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
        trunc_normal_(self.keypoint_token, std=.02)
        if apply_init:
            self.apply(self._init_weights)

    def _make_position_embedding(self, w, h, d_model, pe_type='sine'):
        '''
        d_model: embedding size in transformer encoder
        '''
        assert pe_type in ['none', 'learnable', 'sine', 'sine-full', 'relative']
        if pe_type == 'none':
            self.pos_embedding = None
            print("==> Without any PositionEmbedding~")
        else:
            with torch.no_grad():
                self.pe_h = h
                self.pe_w = w
                length = self.pe_h * self.pe_w
            if pe_type == 'learnable':
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_keypoints, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            elif pe_type == 'sine':
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding_1d(self.num_keypoints, d_model),
                    requires_grad=False)
                print("==> Add Sine PositionEmbedding~")
            elif pe_type == 'relative':
                # self.pos_embedding = nn.Parameter(
                #     self._positional_encoding(self.num_keypoints, d_model), requires_grad=False)
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding_1d(self.num_keypoints, d_model),
                    requires_grad=False)
                self.r_w_bias = nn.Parameter(torch.zeros((self.heads, d_model)), requires_grad=True)
                self.r_r_bias = nn.Parameter(torch.zeros((self.heads, d_model)), requires_grad=True)

    def _positional_encoding(self, num_keypoints, dim, temperature=10000):
        pos_seq = torch.arange(num_keypoints - 1, -1, -1.0, dtype=torch.float32)
        inv_freq = 1 / (temperature ** (torch.arange(0.0, dim, 2.0) / dim))
        sinusoid_inp = torch.outer(pos_seq, inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        return pos_emb[None, :, :]

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
        # # pos = pos.permute(0, 2, 1)
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

    def forward(self, feature, mask=None):
        p = self.patch_size
        # transformer
        x = rearrange(feature, 'b c p1 p2 -> b c (p1 p2)', p1=p[0], p2=p[1])
        x = self.patch_to_embedding(x)

        b, n, _ = x.shape

        # keypoint_tokens = repeat(self.keypoint_token, '() n d -> b n d', b=b)
        # if self.pos_embedding_type in ["sine", "sine-full"]:
        #     x += self.pos_embedding[:, :n]
        #     # x = torch.cat((keypoint_tokens, x), dim=1)
        # else:
        #     # x = torch.cat((keypoint_tokens, x), dim=1)
        #     # x += self.pos_embedding[:, :(n + self.num_keypoints)]
        #     x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x, self.pos_embedding, self.r_w_bias, self.r_r_bias, mask)
        x = self.to_keypoint_token(x[:, 0:self.num_keypoints])
        x = self.mlp_head(x)
        x = rearrange(x, 'b c (p1 p2) -> b c p1 p2', p1=self.heatmap_size[0], p2=self.heatmap_size[1])
        return x


class KITPose(nn.Module):
    def __init__(self, cfg, **kwargs):
        extra = cfg.MODEL.EXTRA
        super(KITPose, self).__init__()

        self.pre_feature = PoseHighResolutionNet(cfg, **kwargs)
        self.channelformer = KITPose_base(
            feature_size=[cfg.MODEL.IMAGE_SIZE[1] // 4, cfg.MODEL.IMAGE_SIZE[0] // 4],
            patch_size=[cfg.MODEL.PATCH_SIZE[1], cfg.MODEL.PATCH_SIZE[0]],
            num_keypoints=cfg.MODEL.NUM_JOINTS, dim=cfg.MODEL.DIM,
            depth=cfg.MODEL.TRANSFORMER_DEPTH, heads=cfg.MODEL.TRANSFORMER_HEADS,
            mlp_dim=cfg.MODEL.DIM * cfg.MODEL.TRANSFORMER_MLP_RATIO,
            attn_type=cfg.MODEL.ATTN_TYPE,
            apply_init=cfg.MODEL.INIT,
            hidden_heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1] * cfg.MODEL.HEATMAP_SIZE[0] // 8,
            heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1] * cfg.MODEL.HEATMAP_SIZE[0],
            heatmap_size=[cfg.MODEL.HEATMAP_SIZE[1], cfg.MODEL.HEATMAP_SIZE[0]],
            pos_embedding_type=cfg.MODEL.POS_EMBEDDING_TYPE)

    def forward(self, x):
        x = self.pre_feature(x)
        x = self.channelformer(x)
        return x

    def init_weights(self, pretrained=''):
        self.pre_feature.init_weights(pretrained)


def get_pose_net(cfg, is_train, **kwargs):
    model = KITPose(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model
