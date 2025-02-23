# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn

from util.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):  # default: num_pos_feats = 128
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask  # (B, H, W)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # (B, H, W) e.g. y_embed[:, 0, :] = 1 y_embed[:, 1, :] = 2. And when there is one false, it will be 0.
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            # y_embed[:, 1:, :]: it stores the non-zero mask's height.
            # y_embed / (y_embed[:, -1:, :] + eps): it will make the element in y_embed between 0 and 1.
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # self.scale = 2 * math.pi
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale  # (bs, 24, 32)

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)  # return [0,self.num_pos_feats-1]
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)# (128, )
        # (bs, 24, 32) -> (bs, 24, 32, 1) -> (bs, 24, 32, 128)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        # pos_y[:, :, :, 0::2].sin(): (bs, 24, 32, 128) -> (bs, 24, 32, 64)
        # pos_y[:, :, :, 1::2].cos(): (bs, 24, 32, 128) -> (bs, 24, 32, 64)
        # torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=4): (bs, 24, 32, 64), (bs, 24, 32, 64) -> (bs, 24, 32, 64, 2)
        # flatten(3): (bs, 24, 32, 64, 2) -> (bs, 24, 32, 128)
        # torch.cat((pos_y, pos_x), dim=3): (bs, 24, 32, 128), (bs, 24, 32, 128) -> (bs, 24, 32, 256)
        # permute(0, 3, 1, 2): (bs, 24, 32, 256) -> (bs, 256, 24, 32)
        # Instead of using `torch.stack([pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()], dim=3)`,
        # it choose stack operation at 4th dimension and flatten them later.
        # it aims to make the sin and cos value alternate. kuhn edited.
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos #  Tensor(bs, 256, 24, 32)
        # # ---------kkuhn-block------------------------------ kuhn: only for testing
        # ss = pos_y[:, :, :, 0::2].sin()
        # s1 = pos_y[:, :, :, 1::2].cos()
        # sss = torch.stack((ss, s1), dim=4)
        # ssss = sss.flatten(3)
        # aa = pos_y[:, :, :, 0::2].sin()
        # a1 = pos_y[:, :, :, 1::2].cos()
        # aaa = torch.stack((aa, a1), dim=4)
        # aaaa = aaa.flatten(3)
        # ww = torch.cat((ssss, aaaa), dim=3)
        # www = ww.permute(0, 3, 1, 2)
        # # ---------kkuhn-block------------------------------


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
