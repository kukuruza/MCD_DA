from __future__ import division

import os
import logging

import torch
from PIL import Image
from torch.autograd import Variable

from argparse import ArgumentParser
from loss import CrossEntropyLoss2d
from models.model_util import get_optimizer, get_full_model  # check_training

parser = ArgumentParser()
parser.add_argument('--net', type=str, default="drn_d_38", help="network structure",
                    choices=['fcn', 'psp', 'segnet', 'fcnvgg',
                              "drn_c_26", "drn_c_42", "drn_c_58", "drn_d_22",
                              "drn_d_38", "drn_d_54", "drn_d_105"])
parser.add_argument('--n_class', default=2, type=int)
parser.add_argument('--input_ch', default=3, type=int)
args = parser.parse_args()

model = get_full_model(net=args.net, res=None, n_class=args.n_class,
                       input_ch=args.input_ch)

image = torch.zeros([2, 3, 64, 64], dtype=torch.float32)
result = model(image)
print (result)

