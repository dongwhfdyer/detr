import os
from pathlib import Path

import cv2
import torch
import einops
import numpy as np
import torch.nn.functional as F
import torchvision
from einops import rearrange
from torch import nn

# #---------kkuhn-block------------------------------ # testing nn.MultiheadAttention
# d_model = 256
# nhead = 8
#
# self_attn = nn.MultiheadAttention(d_model, nhead)
# input = torch.randn(10, 32, d_model)
# target = torch.randn(20, 32, d_model)
# output, attn_output_weights = self_attn(input, target, target)
# print(output.shape)
# print(attn_output_weights.shape)
# print("--------------------------------------------------")
# #---------kkuhn-block------------------------------

# # ---------kkuhn-block------------------------------ test nn.Linear
# dim = 256
# qkv = nn.Linear(dim, dim * 3)
# input = torch.arange(10 * 32 * dim).reshape(10, 32, dim).type(torch.float32)
# output = qkv(input)
#
# equ_mat = torch.arange(dim * dim * 3).reshape(dim, dim * 3).type(torch.float32)
# output_ = input @ equ_mat
# print("--------------------------------------------------")
# # ---------kkuhn-block------------------------------

# ---------kkuhn-block------------------------------ test einops
# https://zhuanlan.zhihu.com/p/342675997
datasetFolder = r"d:\ANewspace\code\DATASETS\rain_data_train_Heavy\norain"
imgs = os.listdir(datasetFolder)[:10]
imgs = [cv2.imread(os.path.join(datasetFolder, img)) for img in imgs]
im0 = imgs[0]
# resize to a size divisible by 2
im0 = cv2.resize(im0, (im0.shape[1] // 30 * 30, im0.shape[0] // 30 * 30))
im_trans = im0.transpose(1, 0, 2)
# im_trans1 = rearrange(im0, '', n1=2, n2=2)
# im_trans1 = rearrange(im0, '(h1 h2) w c -> h1 (h2 w) c', h2=5)
# im_trans1 = rearrange(im0, '(h1 h2) w c -> h1 (w h2) c', h2=2)
im_trans1 = rearrange(im0, '(n1 h1) (n2 h2) c -> (n2 h1) (n1 h2) c', n1=2, n2=2)
# im_trans1 = rearrange(im0, 'h w c -> w h c')
cv2.imshow("im0", im0)
cv2.imshow("im_trans ", im_trans)
cv2.imshow("im_trans1", im_trans1)
cv2.waitKey(0)
# ---------kkuhn-block------------------------------

# #---------kkuhn-block------------------------------ test simple einops
# a = np.arange(6)
# b = rearrange(a, 'n -> () n ()')
# c = rearrange(a, '(n1 n2) -> (n2 n1)', n1=2)
# print("--------------------------------------------------")
# #---------kkuhn-block------------------------------
