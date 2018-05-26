import argparse
import json
import os
from pprint import pprint

import numpy as np
import logging
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor

from models.model_util import get_models, get_full_model
from transform import Scale


class SegmentorSource:

  @staticmethod
  def get_parser():
    parser = argparse.ArgumentParser(description='Trained on synthetic.')
    parser.add_argument('--trained_checkpoint', required=True, type=str)
    return parser

  def __init__(self, args):

    # Load parameters of the network from training arguments.
    print("=> loading checkpoint '{}'".format(args.trained_checkpoint))
    assert os.path.exists(args.trained_checkpoint), args.trained_checkpoint
    checkpoint = torch.load(args.trained_checkpoint)
    train_args = checkpoint["args"]
    print ("----- train args ------")
    pprint(train_args.__dict__, indent=4)
    print ("-" * 50)
    self.input_ch = train_args.input_ch
    assert self.input_ch == 3, self.input_ch
    self.image_shape = tuple([int(x) for x in train_args.train_img_shape])
    print("=> loaded checkpoint '{}'".format(args.trained_checkpoint))

    self.img_transform = Compose([
        Image.fromarray,
        Scale(self.image_shape, Image.BILINEAR),
        ToTensor(),
        Normalize([.485, .456, .406], [.229, .224, .225]),

    ])

    self.model = get_full_model(train_args.net, train_args.res, train_args.n_class, train_args.input_ch)
    self.model.load_state_dict(checkpoint['state_dict'])

    self.model.eval()

    if torch.cuda.is_available():
      self.model.cuda()

    self.add_bg_loss = train_args.add_bg_loss
    self.n_class = train_args.n_class
    print('=> n_class = %d, add_bg_loss = %s' % (self.n_class, self.add_bg_loss))

  def __call__(self, batch):
    imgsnp = batch['image']
    assert len(imgsnp.shape) == 4, imgsnp.shape
    assert imgsnp.shape[3] == self.input_ch

    imgs = torch.stack([self.img_transform(x) for x in imgsnp])

    imgs = Variable(imgs)
    if torch.cuda.is_available():
      imgs = imgs.cuda()

    outputs = self.model(imgs)

    if self.add_bg_loss:
      #preds = outputs.data.max(1)[1].cpu()
      preds = outputs[:, :self.n_class].data.max(1)[1].cpu()
    else:
      preds = outputs[:, :self.n_class - 1].data.max(1)[1].cpu()

    assert self.n_class == 2  # The following replies on self.n_class=2.
    masks = []
    for pred in preds:
      mask = Image.fromarray(np.uint8(pred.numpy() * 255))
      mask = mask.resize((imgsnp.shape[2], imgsnp.shape[1]), Image.NEAREST)
      masknp = np.array(mask) < 128  # Background was 1.
      masks.append(masknp)
    masks = np.array(masks)
    masks = masks[:,:,:,np.newaxis]
    return {'mask': masks}


class SegmentorDA:

  @staticmethod
  def get_parser():
    parser = argparse.ArgumentParser(description='Adaptation from synthetic to real.')
    parser.add_argument('--trained_checkpoint', required=True, type=str)
    parser.add_argument('--uses_one_classifier', action="store_true",
                        help="separate f1, f2")
    parser.add_argument("--use_f2", action="store_true",
                        help='whether you use f2')
    parser.add_argument('--use_ae', action="store_true",
                        help="use ae or not")
    return parser

  def __init__(self, args):

    # Load parameters of the network from training arguments.
    print("=> loading checkpoint '{}'".format(args.trained_checkpoint))
    assert os.path.exists(args.trained_checkpoint), args.trained_checkpoint
    checkpoint = torch.load(args.trained_checkpoint)
    train_args = checkpoint["args"]
    print ("----- train args ------")
    pprint(train_args.__dict__, indent=4)
    print ("-" * 50)
    self.input_ch = train_args.input_ch
    self.image_shape = tuple([int(x) for x in train_args.train_img_shape])
    print("=> loaded checkpoint '{}'".format(args.trained_checkpoint))

    self.img_transform = Compose([
        Image.fromarray,
        Scale(self.image_shape, Image.BILINEAR),
        ToTensor(),
        Normalize([.485, .456, .406], [.229, .224, .225]),

    ])

    try:
      self.G, self.F1, self.F2 = get_models(net_name=train_args.net, res=train_args.res, input_ch=train_args.input_ch,
                            n_class=train_args.n_class,
                            method=train_args.method, is_data_parallel=train_args.is_data_parallel,use_ae=args.use_ae)
    except AttributeError:
      self.G, self.F1, self.F2 = get_models(net_name=train_args.net, res=train_args.res, input_ch=train_args.input_ch,
                            n_class=train_args.n_class,
                            method="MCD", is_data_parallel=False)

    self.G.load_state_dict(checkpoint['g_state_dict'])
    self.F1.load_state_dict(checkpoint['f1_state_dict'])

    if args.use_f2:
      self.F2.load_state_dict(checkpoint['f2_state_dict'])
    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.trained_checkpoint, checkpoint['epoch']))

    self.G.eval()
    self.F1.eval()
    self.F2.eval()

    if torch.cuda.is_available():
      self.G.cuda()
      self.F1.cuda()
      self.F2.cuda()

    self.use_f2 = args.use_f2

    self.add_bg_loss = train_args.add_bg_loss
    self.n_class = train_args.n_class
    print('=> n_class = %d, add_bg_loss = %s' % (self.n_class, self.add_bg_loss))

  def __call__(self, batch):
    imgsnp = batch['image']
    assert len(imgsnp.shape) == 4, imgsnp.shape
    assert imgsnp.shape[3] == self.input_ch

    imgs = torch.stack([self.img_transform(x) for x in imgsnp])

    imgs = Variable(imgs)
    if torch.cuda.is_available():
      imgs = imgs.cuda()

    feature = self.G(imgs)
    outputs = self.F1(feature)

    if self.use_f2:
      outputs += self.F2(feature)

    if self.add_bg_loss:
      preds = outputs[:, :self.n_class].data.max(1)[1].cpu()
    else:
      preds = outputs[:, :self.n_class - 1].data.max(1)[1].cpu()

    assert self.n_class == 2  # The following replies on self.n_class=2.
    masks = []
    for pred in preds:
      mask = Image.fromarray(np.uint8(pred.numpy() * 255))
      mask = mask.resize((imgsnp.shape[2], imgsnp.shape[1]), Image.NEAREST)
      masknp = np.array(mask) < 128  # Background was 1.
      masks.append(masknp)
    masks = np.array(masks)
    masks = masks[:,:,:,np.newaxis]
    return {'mask': masks}


if __name__ == '__main__':

  parser = Segmentor.get_parser()
  parser.add_argument('--logging', type=int, choices=[10,20,30,40], default=20)
  args = parser.parse_args()
  logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')
  
  segmentor = Segmentor(args)
  images = np.random.randint(low=0, high=255, size=(10, 100, 100, 3), dtype=np.uint8)
  outputs = segmentor({'image': images})
  masks = outputs['mask']
  print masks.shape

