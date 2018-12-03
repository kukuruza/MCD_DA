import os
import logging
import argparse
from datetime import datetime
import time

import torch
from PIL import Image
from tensorboard_logger import configure, log_value
from torch.autograd import Variable
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor

from datasets import get_dataset
from transform import ReLabel, ToLabel, Scale, RandomSizedCrop, RandomHorizontalFlip, RandomRotation

parser = argparse.ArgumentParser('Benchmark speed of datasets.')
parser.add_argument('src_dataset', choices=['citycam'])
parser.add_argument('--split', required=True)
parser.add_argument('--batch_size', type=int, required=True)
parser.add_argument('--num_workers', type=int, required=True)
parser.add_argument('--max_iter', type=int)
parser.add_argument('--logging', type=int, choices=[11,20,30,40], default=20)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--augment', action='store_true')
parser.add_argument('--train_img_shape', type=int, nargs='+', default=[64, 64])
args = parser.parse_args()

FORMAT = '[%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s'
logging.basicConfig(level=args.logging, format=FORMAT)

train_img_shape = tuple([int(x) for x in args.train_img_shape])

img_transform_list = [
    Scale(train_img_shape, Image.BILINEAR),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225])
]

if args.augment:
    aug_list = [
        RandomRotation(),
        # RandomVerticalFlip(), # non-realistic
        RandomHorizontalFlip(),
        RandomSizedCrop()
    ]
    img_transform_list = aug_list + img_transform_list

img_transform = Compose(img_transform_list)

label_transform = Compose([
    Scale(train_img_shape, Image.NEAREST),
    ToLabel(),
    ReLabel(255, 1),
])

src_dataset = get_dataset(dataset_name=args.src_dataset, split=args.split, img_transform=img_transform,
                          label_transform=label_transform, test=False, input_ch=3,
                          keys_dict={'image': 'image', 'image_original': 'image_original', 'mask': 'mask', 'yaw': 'yaw', 'url': 'url'})

kwargs = {'pin_memory': True} if torch.cuda.is_available() else {}
start = datetime.now()
loader = torch.utils.data.DataLoader(src_dataset, batch_size=args.batch_size,
        shuffle=args.shuffle, num_workers=args.num_workers, **kwargs)
print('Time: created the dataset: %.4f sec.' % (datetime.now() - start).total_seconds())

accum = 0
start = datetime.now()
for ibatch, batch in enumerate(loader):
    if args.max_iter is not None and ibatch > args.max_iter:
        break

    entries = [batch[name] for name in ['image', 'mask', 'yaw']]
    if torch.cuda.is_available():
        for entry in entries:
            entry = entry.cuda()

    if ibatch == 1:
        print('Time: batch: %.4f sec.' % (datetime.now() - start).total_seconds())
    accum += (datetime.now() - start).total_seconds()
    start = datetime.now()

print('Time: load from dataset: %.4f' % accum)

