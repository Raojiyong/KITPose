import copy

import logging
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn
from timm.models.layers.weight_init import trunc_normal_
from .hr_base import HRNET_base
import math

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


# original version
class MHAttn(nn.Module):
    def __init__(self, dim, heads=8, dropout=0., num_keypoints=None, scale_with_head=False):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5 if scale_with_head else dim ** -0.5

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )
        self.num_keypoints = num_keypoints

    def forward(self, qkv, attn_mask=None, average_attn_weights=False):
        b, n, _, h = *qkv[0].shape, self.heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        attn_mask_value = -torch.finfo(dots.dtype).max
        if attn_mask is not None:
            attn_mask = F.pad(attn_mask.flatten(1), (1, 0), value=True)
            assert attn_mask.shape[-1] == dots.shape[-1], 'attn_mask has incorrect dimensions'
            attn_mask = attn_mask[:, None, :] * attn_mask[:, :, None]
            dots.attn_masked_fill_(~attn_mask, attn_mask_value)
            del attn_mask

        attn = dots.softmax(dim=-1)
        if average_attn_weights:
            attn = attn.sum(dim=1, keepdim=True) / h

        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out, attn


class PoseEncoderLayer(nn.Module):
    def __init__(self, dim, heads, dropout, mlp_dim, num_keypoints=None, scale_with_head=False):
        super().__init__()
        self.self_attn = MHAttn(dim, heads=heads, dropout=dropout, num_keypoints=num_keypoints,
                                scale_with_head=scale_with_head)
        self.multihead_attn = MHAttn(dim, heads=heads, dropout=dropout, num_keypoints=num_keypoints,
                                     scale_with_head=scale_with_head)
        self.ffn = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, mask=None, kpt_pos=None, pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, kpt_pos)
        qkv = (q, k, tgt2)
        tgt2 = self.self_attn(qkv, attn_mask=mask)[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        qkv = (self.with_pos_embed(tgt2, kpt_pos), self.with_pos_embed(memory, pos), memory)
        tgt2 = self.multihead_attn(qkv, attn_mask=mask)[0]
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.ffn(tgt2)
        tgt = tgt + self.dropout3(tgt2)
        return tgt


class PoseEncoder(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, num_keypoints=None, scale_with_head=True):
        super().__init__()
        self.layers = nn.ModuleList([
            PoseEncoderLayer(dim, heads, dropout, mlp_dim, num_keypoints, scale_with_head)
            for _ in range(depth)
        ])

    def forward(self, tgt, memory, kpt_embed, mask=None, pos=None):
        output = tgt
        for layer in self.layers:
            output = layer(output, memory, mask=mask, kpt_pos=kpt_embed, pos=pos)
        return output


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout, num_keypoints=None, scale_with_head=True):
        super().__init__()
        self.encoder = PoseEncoder(dim, depth, heads, mlp_dim, dropout, num_keypoints, scale_with_head=scale_with_head)
        self.dim = dim
        self.heads = heads

    def forward(self, src, mask, kpt_embed, pos_embed):
        tgt = torch.randn_like(kpt_embed)
        out = self.encoder(tgt, src, kpt_embed, pos=pos_embed, mask=mask)
        return out


# def _get_clones(module, N):
#     return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class KPTR(nn.Module):
    def __init__(self, *, feature_size, patch_size, num_keypoints, dim, depth, heads, mlp_dim, apply_init=False,
                 apply_multi=True, hidden_heatmap_dim=64 * 32, heatmap_dim=64 * 64, heatmap_size=[64, 64], channels=3,
                 dropout=0., pos_embedding_type="learnable"):
        super().__init__()
        assert isinstance(feature_size, list) and isinstance(patch_size,
                                                             list), 'image_size and patch_size should be list'
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[
            1] == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (feature_size[0] // (patch_size[0])) * (feature_size[1] // (patch_size[1]))
        patch_dim = patch_size[0] * patch_size[1] * channels
        assert pos_embedding_type in ['none', 'sine', 'learnable', 'sine-full']

        self.inplanes = 64
        self.patch_size = patch_size
        self.heatmap_size = heatmap_size
        self.num_keypoints = num_keypoints
        self.num_patches = num_patches
        self.pos_embedding_type = pos_embedding_type

        self.keypoint_token = nn.Parameter(torch.zeros(1, self.num_keypoints, dim))
        h, w = feature_size[0] // (self.patch_size[0]), feature_size[1] // (self.patch_size[1])
        self._make_position_embedding(w, h, dim, pos_embedding_type)

        self.patch_to_embedding = nn.Linear(patch_dim, dim)

        # transformer
        self.transformer = Transformer(dim, depth, heads, mlp_dim, dropout, num_keypoints=num_keypoints,
                                       scale_with_head=True)

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
                self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
                trunc_normal_(self.pos_embedding, std=.02)
                print("==> Add Learnable PositionEmbedding~")
            else:
                self.pos_embedding = nn.Parameter(
                    self._make_sine_position_embedding_2d(d_model),
                    requires_grad=False)
                # self.pos_embedding = nn.Parameter(
                #     self._make_sine_position_embedding_1d(self.num_keypoints, d_model),
                #     requires_grad=False)
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
        x = rearrange(feature, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p[0], p2=p[1])
        x = self.patch_to_embedding(x)

        b, n, _ = x.shape

        keypoint_tokens = repeat(self.keypoint_token, '() n d -> b n d', b=b)

        x = self.transformer(x, mask, keypoint_tokens, self.pos_embedding)
        x = self.to_keypoint_token(x)
        x = self.mlp_head(x)
        x = rearrange(x, 'b c (p1 p2) -> b c p1 p2', p1=self.heatmap_size[0], p2=self.heatmap_size[1])
        return x


class KPTR_B(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(KPTR_B, self).__init__()
        self.pre_feature = HRNET_base(cfg, **kwargs)
        self.channelformer = KPTR(feature_size=[cfg.MODEL.IMAGE_SIZE[1] // 4, cfg.MODEL.IMAGE_SIZE[0] // 4],
                                  patch_size=[cfg.MODEL.PATCH_SIZE[1], cfg.MODEL.PATCH_SIZE[0]],
                                  num_keypoints=cfg.MODEL.NUM_JOINTS, dim=cfg.MODEL.DIM,
                                  depth=cfg.MODEL.TRANSFORMER_DEPTH, heads=cfg.MODEL.TRANSFORMER_HEADS,
                                  mlp_dim=cfg.MODEL.DIM * cfg.MODEL.TRANSFORMER_MLP_RATIO,
                                  apply_init=cfg.MODEL.INIT,
                                  channels=cfg.MODEL.BASE_CHANNEL,
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
    model = KPTR_B(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model

# @register_model
# def kptr_b(cfg, is_train, **kwargs):
#     model = KPTR(feature_size=[cfg.MODEL.IMAGE_SIZE[1] // 4, cfg.MODEL.IMAGE_SIZE[0] // 4],
#                  patch_size=[cfg.MODEL.PATCH_SIZE[1], cfg.MODEL.PATCH_SIZE[0]],
#                  num_keypoints=cfg.MODEL.NUM_JOINTS, dim=cfg.MODEL.DIM,
#                  depth=cfg.MODEL.TRANSFORMER_DEPTH, heads=cfg.MODEL.TRANSFORMER_HEADS,
#                  mlp_dim=cfg.MODEL.DIM * cfg.MODEL.TRANSFORMER_MLP_RATIO,
#                  apply_init=cfg.MODEL.INIT,
#                  hidden_heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1] * cfg.MODEL.HEATMAP_SIZE[0] // 8,
#                  heatmap_dim=cfg.MODEL.HEATMAP_SIZE[1] * cfg.MODEL.HEATMAP_SIZE[0],
#                  heatmap_size=[cfg.MODEL.HEATMAP_SIZE[1], cfg.MODEL.HEATMAP_SIZE[0]],
#                  pos_embedding_type=cfg.MODEL.POS_EMBEDDING_TYPE)
#
#     if is_train and cfg['MODEL']['INIT_WEIGHTS']:
#         model.init_weights(cfg['MODEL']['PRETRAINED'])
#
#     return model
