from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers.weight_init import trunc_normal_
import numpy as np
import math
# from .kitpose_base import KITPose_base
from .fast_keans import batch_fast_kmedoids_with_split
from .pose_hrnet import PoseHighResolutionNetMOE, Bottleneck, BasicBlock
from .hr_base import HRNET_base

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


# def index_points(points, idx):
#     device = points.device
#     B = points.shape[0]
#     view_shape = list(idx.shape)
#     view_shape[1:] = [1] * (len(view_shape) - 1)
#     repeat_shape = list(idx.shape)
#     repeat_shape[0] = 1
#     batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
#     new_points = points[batch_indices, idx, :]
#     return new_points
#
#
# def cluster_dpc_knn(tokens, cluster_num, k=5, token_mask=None):
#     with torch.no_grad():
#         x = tokens
#         B, N, C = x.shape
#
#         dist_matrix = torch.cdist(x, x) / (C ** 0.5)
#
#         if token_mask is not None:
#             token_mask = token_mask > 0
#             dist_matrix = dist_matrix * token_mask[:, None, :] + \
#                           (dist_matrix.max() + 1) * (~token_mask[:, None, :])
#
#         # get local density
#         dist_nearest, index_nearest = torch.topk(dist_matrix, k=k, dim=-1, largest=False)
#
#         density = (-(dist_nearest ** 2).mean(dim=-1)).exp()
#         # add a little noise to ensure no tokens have the same density
#         density = density + torch.rand(
#             density.shape, device=density.device, dtype=density.dtype
#         ) * 1e-6
#
#         if token_mask is not None:
#             # the density of empty token should be 0
#             density = density * token_mask
#
#         # get distance indicator
#         mask = density[:, None, :] > density[:, :, None]
#         mask = mask.type(x.dtype)
#         dist_max = dist_matrix.flatten(1).max(dim=-1)[0][:, None, None]
#         dist, index_parent = (dist_matrix * mask + dist_max * (1 - mask)).min(dim=-1)
#
#         # select clustering center according to score
#         score = dist * density
#         _, index_down = torch.topk(score, k=cluster_num, dim=-1)
#
#         dist_matrix = index_points(dist_matrix, index_down)
#
#         idx_cluster = dist_matrix.argmin(dim=1)
#
#         idx_batch = torch.arange(B, device=x.device)[:, None].expand(B, cluster_num)
#         idx_tmp = torch.arange(cluster_num, device=x.device)[None, :].expand(B, cluster_num)
#         idx_cluster[idx_batch.reshape(-1), index_down.reshape(-1)] = idx_tmp.reshape(-1)
#
#     return idx_cluster, cluster_num
#
#
# def merge_tokens(tokens, idx_cluster, cluster_num, token_weight=None):
#     B, N, C = tokens.shape
#     if token_weight is None:
#         token_weight = tokens.new_ones(B, N, 1)
#
#     idx_batch = torch.arange(B, device=tokens.device)[:, None]
#     idx = idx_cluster + idx_batch * cluster_num
#
#     all_weight = token_weight.new_zeros(B * cluster_num, 1)
#     all_weight.index_add_(dim=0, index=idx.reshape(B * N),
#                           source=token_weight.reshape(B * N, 1))
#     all_weight = all_weight + 1e-6
#     norm_weight = token_weight / all_weight[idx]
#
#     # average token features
#     x_merged = tokens.new_zeros(B * cluster_num, C)
#     source = tokens * norm_weight
#     x_merged.index_add_(dim=0, index=idx.reshape(B * N),
#                         source=source.reshape(B * N, C).type(tokens.dtype))
#     x_merged = x_merged.reshape(B, cluster_num, C)
#     return x_merged


# class DWConv(nn.Module):
#     def __init__(self, dim=17):
#         super(DWConv, self).__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
#
#     def forward(self, x):
#         B, N, C = x.shape
#         HW = int(math.sqrt(C))
#         # x = rearrange(x,, 'b n (h w) -> b n h w', h = HW, w = HW)
#         x = x.view(B, N, HW, HW)
#         x = self.dwconv(x)
#         x = x.flatten(2)
#         return x


# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim, in_channels=17, act_layer=nn.GELU, dropout=0.):
#         super().__init__()
#         self.fc1 = nn.Sequential(nn.Linear(dim, hidden_dim),
#                                  act_layer())
#         self.drop = nn.Dropout(dropout)
#         self.act = act_layer()
#         self.dwconv = nn.Sequential(DWConv(in_channels),
#                                     act_layer())
#         self.fc2 = nn.Linear(hidden_dim, dim)
#
#     def forward(self, x):
#         B, N, C = x.shape
#         x = self.fc1(x)
#         # x = self.act(x + self.dwconv(x))
#         x = self.dwconv(x)
#         x = self.drop(x)
#         x = self.fc2(x)
#         x = self.drop(x)
#         return x


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


class MoEFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim=None, num_expert=1, dropout=0., part_dim=256):
        super().__init__()
        out_dim = out_dim or dim
        hidden_dim = hidden_dim or dim
        self.part_dim = part_dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim - part_dim)
        self.drop = nn.Dropout(dropout)

        self.num_expert = num_expert
        experts = []

        for i in range(num_expert):
            experts.append(nn.Linear(hidden_dim, part_dim))
        self.experts = nn.ModuleList(experts)

    def forward(self, x, indices):
        expert_x = torch.zeros_like(x[:, :, -self.part_dim:], device=x.device, dtype=x.dtype)

        x = self.fc1(x)
        x = self.act(x)
        shared_x = self.fc2(x)
        indices = indices.view(-1, 1, 1)

        for i in range(self.num_expert):
            selectedIndex = (indices == i)
            current_x = self.experts[i](x) * selectedIndex
            expert_x = expert_x + current_x

        x = torch.cat([shared_x, expert_x], dim=-1)
        return x


# class InnerAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads, dropout=0., kdim=None, vdim=None):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.kdim = kdim if kdim is not None else embed_dim
#         self.vdim = vdim if vdim is not None else embed_dim
#
#         self.num_heads = num_heads
#         self.dropout = nn.Dropout(dropout)
#         self.query_proj = nn.Linear(self.kdim, self.kdim, bias=True)
#         self.key_proj = nn.Linear(self.kdim, self.kdim, bias=True)
#         self.out_proj = nn.Linear(self.vdim, self.vdim, bias=True)
#         self.scale = embed_dim ** -0.5
#
#     def forward(self, query, key, value):
#         q, k = self.query_proj(query), self.key_proj(key)
#         v = value
#         dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
#         attn = dots.softmax(dim=-1)
#         attn = self.dropout(attn)
#         out = torch.einsum('bij,bjd->bid', attn, v)
#         residual = out
#         out = self.out_proj(out)
#         out = residual + out
#         return out, attn
#
#
# class CorrAttention(nn.Module):
#     def __init__(self, num_heads, dropout, match_dim, feat_size):
#         super().__init__()
#         self.match_dim = match_dim
#         self.feat_size = feat_size
#         self.corr_proj = nn.Linear(self.feat_size, self.match_dim)
#         self.inner_attn = InnerAttention(self.match_dim, num_heads, dropout=dropout, vdim=self.feat_size)
#         self.feat_norm1 = nn.LayerNorm(self.match_dim)
#         self.feat_norm2 = nn.LayerNorm(self.feat_size)
#         self.dropout = nn.Dropout(dropout)
#         self.num_heads = num_heads
#
#     def forward(self, corr_map, pos_emb):
#         batch_size = pos_emb.shape[0]
#         pos_emb = torch.repeat_interleave(pos_emb, self.num_heads, dim=1).reshape(-1, self.feat_size, self.match_dim)
#         # from the perspective of keys
#         corr_map = corr_map.transpose(-2, -1).reshape(-1, self.feat_size, self.feat_size)
#         q = k = (self.feat_norm1(self.corr_proj(corr_map)) + pos_emb)
#         corr_map1 = self.inner_attn(q, k, value=self.feat_norm2(corr_map))[0]
#         corr_map = self.dropout(corr_map1)
#         corr_map = corr_map.transpose(-2, -1)
#         corr_map = corr_map.reshape(self.num_heads * batch_size, self.feat_size, -1)
#         return corr_map


class Attention(nn.Module):
    def __init__(self, dim, inner_dim, heads=8, dropout=0., scale_with_head=False, aia_mode=False):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5 if scale_with_head else dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=True)
        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        # self.num_keypoints = num_keypoints
        self.aia_mode = aia_mode
        # self.corr_attn = CorrAttention(num_heads=heads, dropout=dropout, match_dim=inner_dim, feat_size=num_keypoints)

    # @get_local('attn')
    def forward(self, x, mask=None, pos_embed=None, inner_pos_embed=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = -torch.finfo(dots.dtype).max

        if mask is not None:
            # mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, mask_value)
            del mask

        # if self.aia_mode:
        #     corr_map = dots
        #     corr_map = self.corr_attn(corr_map, inner_pos_embed).reshape(b, h, n, -1)
        #     dots = corr_map + dots

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out, attn


class PoseEncoderLayer(nn.Module):
    def __init__(self, dim, inner_dim, heads, dropout, mlp_dim, all_attn=False,
                 scale_with_head=False, aia_mode=False,
                 num_expert=1, part_dim=256):
        super().__init__()
        self.attn = Attention(dim, inner_dim, heads=heads, dropout=dropout,
                              scale_with_head=scale_with_head, aia_mode=aia_mode)
        self.ffn = MoEFeedForward(dim, mlp_dim, num_expert=num_expert, dropout=dropout, part_dim=part_dim)
        # self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, mask=None, pos=None, inner_pos=None, indices=None):
        residual = x
        x = self.norm1(x)
        x, attn = self.attn(x, mask, pos, inner_pos)
        x += residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x, indices)
        out = x + residual
        return out, attn


class Transformer(nn.Module):
    def __init__(self, dim, inner_dim, depth, heads, mlp_dim, dropout, all_attn=False,
                 scale_with_head=False, aia_mode=False,
                 num_expert=1, part_dim=None):
        super().__init__()
        self.all_attn = all_attn
        # self.num_keypoints = num_keypoints
        # self.num_bp = num_bp
        self.part_dim = part_dim
        self.layers = nn.ModuleList([
            PoseEncoderLayer(dim, inner_dim, heads, dropout, mlp_dim, all_attn, scale_with_head,
                             aia_mode, num_expert=num_expert, part_dim=part_dim)
            for _ in range(depth)
        ])

    def forward(self, x, mask=None, pos=None, inner_pos=None, indices=None):
        # normal cnnect
        for idx, layer in enumerate(self.layers):
            if idx > 0 and self.all_attn:
                x[:, self.num_bp:] += pos
            x, attn_weights = layer(x, mask=mask, pos=pos, inner_pos=inner_pos, indices=indices)
            att = attn_weights
        # dense connect
        # features = [x]
        # for idx, layer in enumerate(self.layers):
        #     feature = torch.stack(features).sum(dim=0)
        #     if idx > 0 and self.all_attn:
        #         feature[:, self.num_bp:] += pos
        #     new_feature, attn_weights = layer(feature, mask=mask, pos=pos, inner_pos=inner_pos)
        #     features.append(new_feature)
        # out = torch.stack(features).sum(dim=0)
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
    def __init__(self, *, feature_size, kpt_size, num_keypoints, num_bp, dim, inner_dim,
                 depth, heads, mlp_dim, apply_init=False, apply_multi=True, hidden_heatmap_dim=64 * 32, heatmap_dim=64 * 64,
                 heatmap_size=[64, 64], channels=32, dropout=0., emb_dropout=0., pos_embedding_type="learnable",
                 aia_mode=False, num_expert=1, part_dim=None):
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
        # self.max_num_keypoint = max_num_keypoints
        self.num_bp = num_bp
        # self.max_num_bp = max_num_bp
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type
        self.all_attn = (self.pos_embedding_type == "sine-full")
        self.aia_mode = aia_mode
        self.num_expert = num_expert

        # self.bp_embeddings = nn.Parameter(torch.zeros(1, self.num_bp, dim))
        h, w = feature_size[0] // (self.patch_size[0]), feature_size[1] // (self.patch_size[1])
        self._make_position_embedding(w, h, dim, inner_dim, pos_embedding_type)

        self.kpt_to_embedding = nn.Linear(kpt_dim, dim)
        self.bp_to_embeddings = nn.Linear(kpt_dim, dim)
        self.dropout = nn.Dropout(emb_dropout)

        # transformer
        self.transformer = Transformer(dim, inner_dim, depth, heads, mlp_dim, dropout, all_attn=self.all_attn,
                                       scale_with_head=False, aia_mode=aia_mode,
                                       num_expert=num_expert, part_dim=part_dim)

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
        # trunc_normal_(self.bp_embeddings, std=.02)
        self.prompt_learners = nn.ModuleList([
            PromptLearner(i, heatmap_dim, dim, self.inplanes)
            for i in num_bp
        ])

        # associate_prompt_learner = []
        # for i in range(1, num_expert):
        #     associate_prompt_learner.append(PromptLearner(num_bp[i], heatmap_dim, dim, self.inplanes))
        # self.associate_prompt_learner = nn.ModuleList(associate_prompt_learner)

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
                # self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + self.num_keypoints, d_model))
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_keypoints[0] + self.num_bp, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                # self.pos_embedding = nn.Parameter(
                #     self._make_sine_position_embedding_2d(d_model),
                #     requires_grad=False)
                self.pos_embedding = [
                    nn.Parameter(self._make_sine_position_embedding_1d(i, d_model), requires_grad=False)
                    for i in self.num_keypoints
                ]
                # if self.aia_mode:
                #     self.inner_pos_embedding = nn.Parameter(
                #         self._make_sine_position_embedding_1d(self.num_keypoints + self.num_bp, d_inner),
                #         requires_grad=False)
                # else:
                #     self.inner_pos_embedding = None
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
        # pos = pos.permute(0, 2, 1)
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

    # def _get_bp(self, x, cluster_num):
    #     assign, mediods_ids = batch_fast_kmedoids_with_split(x, cluster_num,
    #                                                          distance='euclidean', threshold=1e-6,
    #                                                          iter_limit=40,
    #                                                          id_sort=True,
    #                                                          norm_p=2.0,
    #                                                          split_size=8,
    #                                                          pre_norm=True)
    #     bp_emb_list = []
    #     for i in range(cluster_num):
    #         # [B, cluster, 1]
    #         bp_mask = (assign == i).unsqueeze(-1)
    #         # [B, 1, width]
    #         x_tmp_tmp = torch.sum(x * bp_mask, dim=1, keepdim=True) / torch.sum(
    #             bp_mask.float(), dim=1, keepdim=True)
    #         bp_emb_list.append(x_tmp_tmp)
    #     # [B x T_new, cluster, width]
    #     bp_emb_tmp = torch.cat(bp_emb_list, dim=1)
    #     return bp_emb_tmp

    def forward(self, all_feature, kpt_feature, mask=None, indices=None):
        p = self.patch_size
        # transformer
        kpt_feature = rearrange(kpt_feature, 'b c p1 p2 -> b c (p1 p2)', p1=p[0], p2=p[1])
        # kpt_feature = kpt_feature[:, :self.num_keypoints]
        # all_feature = rearrange(all_feature, 'b c p1 p2 -> b c (p1 p2)', p1=p[0], p2=p[1])
        x = self.kpt_to_embedding(kpt_feature)
        # bp_embeddings = self.bp_to_embeddings(all_feature)

        b, n, d = x.shape

        cluster_num = self.num_bp
        # batch_index = torch.arange(kpt_feature.shape[0], dtype=torch.long, device=kpt_feature.device).unsqueeze(-1)
        # idx_cluster, cluster_num = cluster_dpc_knn(all_feature, cluster_num, 2)
        # bp_embeddings = merge_tokens(all_feature, idx_cluster, cluster_num)
        assign, mediods_ids = batch_fast_kmedoids_with_split(kpt_feature, cluster_num,
                                                             distance='euclidean', threshold=1e-6,
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
        # for i in range(self.num_expert):
        #     idx_select = (indices == i)
        #     cluster_num = self.num_bp[i]
        #     bp_emb_tmp = self._get_bp(x, cluster_num)
        #     if self.max_num_bp > cluster_num:
        #         bp_embeddings = torch.cat(
        #             (bp_emb_tmp, torch.zeros((b, self.max_num_bp - cluster_num, d), dtype=bp_emb_tmp.dtype, requires_grad=False)),
        #             dim=1)
        bp_embeddings = self.prompt_learners[0](all_feature, bp_emb_tmp)

        # bp_embeddings = bp_embeddings.view(b, 1, self.max_num_bp, d)
        # for i in range(1, len(self.num_bp)):
        #     cluster_num = self.num_bp[i]
        #     bp_emb_tmp = self._get_bp(x, cluster_num)
        #     bp_embedding = self.associate_prompt_learner[i - 1](all_feature, bp_emb_tmp)
        #     if self.max_num_bp > cluster_num:
        #         bp_embedding = torch.cat(
        #             (bp_embedding, torch.zeros((b, self.max_num_bp - cluster_num, d), dtype=bp_embedding.dtype, requires_grad=False)),
        #             dim=1)
        #     bp_embedding = bp_embedding.view(b, 1, self.max_num_bp, d)
        #     bp_embeddings = torch.cat((bp_embeddings, bp_embedding), dim=1)

        if self.pos_embedding_type in ["sine", "sine-full"]:
            x += self.pos_embedding[:, :n]
            x = torch.cat((bp_embeddings, x), dim=1)
        else:
            x = torch.cat((bp_embeddings, x), dim=1)
            x += self.pos_embedding[:, :(n + self.num_keypoints)]
            # x += self.pos_embedding[:, :n]
        x = self.dropout(x)

        x = self.transformer(x, mask, self.pos_embedding, indices=indices)
        x = self.mlp_head(x)
        x = rearrange(x, 'b c (p1 p2) -> b c p1 p2', p1=self.heatmap_size[0], p2=self.heatmap_size[1])
        bp_embeddings, x = \
            self.to_keypoint_token(x[:, :self.num_bp]), self.to_keypoint_token(x[:, self.num_bp:])
        return bp_embeddings, x


class KITPose(nn.Module):
    def __init__(self, cfg, **kwargs):
        extra = cfg.MODEL.EXTRA
        super(KITPose, self).__init__()
        self.pre_feature = PoseHighResolutionNetMOE(cfg, **kwargs)
        # self.pre_feature = HRNET_base(cfg, map=True, **kwargs)
        self.channelformer = KITPose_base(
            feature_size=[cfg.MODEL.IMAGE_SIZE[1] // 4, cfg.MODEL.IMAGE_SIZE[0] // 4],
            kpt_size=[cfg.MODEL.PATCH_SIZE[1], cfg.MODEL.PATCH_SIZE[0]],
            num_keypoints=cfg.MODEL.MAX_NUM_JOINTS, num_bp=cfg.MODEL.MAX_NUM_BP,
            # max_num_keypoints=cfg.MODEL.MAX_NUM_JOINTS, max_num_bp=cfg.MODEL.MAX_NUM_BP,
            dim=cfg.MODEL.DIM, inner_dim=cfg.MODEL.INNER_DIM,
            channels=extra.STAGE2.NUM_CHANNELS[0],
            depth=cfg.MODEL.TRANSFORMER_DEPTH, heads=cfg.MODEL.TRANSFORMER_HEADS,
            mlp_dim=cfg.MODEL.DIM * cfg.MODEL.TRANSFORMER_MLP_RATIO,
            apply_init=cfg.MODEL.INIT,
            hidden_heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1] * cfg.MODEL.HEATMAP_SIZE[0] // 8,
            heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1] * cfg.MODEL.HEATMAP_SIZE[0],
            heatmap_size=[cfg.MODEL.HEATMAP_SIZE[1], cfg.MODEL.HEATMAP_SIZE[0]],
            pos_embedding_type=cfg.MODEL.POS_EMBEDDING_TYPE,
            aia_mode=cfg.MODEL.AIA_MODE,
            num_expert=cfg.MODEL.NUM_EXPERTS,
            part_dim=cfg.MODEL.PART_DIM,
        )
        #
        # associate_former = []
        # for i in range(1, cfg.MODEL.NUM_EXPERTS):
        #     associate_former.append(KITPose_base(
        #         feature_size=[cfg.MODEL.IMAGE_SIZE[1] // 4, cfg.MODEL.IMAGE_SIZE[0] // 4],
        #         kpt_size=[cfg.MODEL.PATCH_SIZE[1], cfg.MODEL.PATCH_SIZE[0]],
        #         num_keypoints=cfg.MODEL.NUM_JOINTS[i], num_bp=cfg.MODEL.NUM_BP[i],
        #         dim=cfg.MODEL.DIM, inner_dim=cfg.MODEL.INNER_DIM,
        #         channels=extra.STAGE2.NUM_CHANNELS[0],
        #         depth=cfg.MODEL.TRANSFORMER_DEPTH, heads=cfg.MODEL.TRANSFORMER_HEADS,
        #         mlp_dim=cfg.MODEL.DIM * cfg.MODEL.TRANSFORMER_MLP_RATIO,
        #         apply_init=cfg.MODEL.INIT,
        #         hidden_heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1] * cfg.MODEL.HEATMAP_SIZE[0] // 8,
        #         heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1] * cfg.MODEL.HEATMAP_SIZE[0],
        #         heatmap_size=[cfg.MODEL.HEATMAP_SIZE[1], cfg.MODEL.HEATMAP_SIZE[0]],
        #         pos_embedding_type=cfg.MODEL.POS_EMBEDDING_TYPE,
        #         aia_mode=cfg.MODEL.AIA_MODE,
        #         num_expert=cfg.MODEL.NUM_EXPERTS,
        #         part_dim=cfg.MODEL.PART_DIM,
        #     ))
        # self.associate_formers = nn.ModuleList(associate_former)

    def forward(self, x, indices=None, mask=None):
        all_feats, kpt_feats = self.pre_feature(x, indices=indices)
        bp, out = self.channelformer(all_feats, kpt_feats, mask=mask, indices=indices)
        return kpt_feats, bp, out

    def init_weights(self, pretrained=''):
        self.pre_feature.init_weights(pretrained)


def get_pose_net(cfg, is_train, **kwargs):
    model = KITPose(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model
