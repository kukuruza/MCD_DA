import os, sys, os.path as op
import argparse
import torch
from torch.autograd import Variable
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor
sys.path.insert(0, op.join(os.getenv('HOME'), 'src/MCD_DA/segmentation'))
from transform import Scale


class ClassifierYawDA:

  @staticmethod
  def get_parser():
    parser = argparse.ArgumentParser(description='Classification yaw with adaptation.')
    parser.add_argument('--yaw_checkpoint_dir', required=True, type=str)
    parser.add_argument('--yaw_epoch', required=True, type=str)
    parser.add_argument('--train_img_shape', nargs='+', required=True, type=int)
    return parser

  def __init__(self, args):

    train_img_shape = tuple(args.train_img_shape)
    self.img_transform = Compose([
        Image.fromarray,
        Scale(train_img_shape, Image.BILINEAR),
        ToTensor(),
        Normalize([.485, .456, .406], [.229, .224, .225])
    ])

    self.G = torch.load(op.join(args.yaw_checkpoint_dir, 'citycam_to_citycam_model_epoch%s_G.pt' % args.yaw_epoch))
    self.C1 = torch.load(op.join(args.yaw_checkpoint_dir, 'citycam_to_citycam_model_epoch%s_C1.pt' % args.yaw_epoch))
    self.C2 = torch.load(op.join(args.yaw_checkpoint_dir, 'citycam_to_citycam_model_epoch%s_C2.pt' % args.yaw_epoch))
    print('load finished!')

    self.G.cuda()
    self.C1.cuda()
    self.C2.cuda()

    self.G.eval()
    self.C1.eval()
    self.C2.eval()

    self.input_ch = 3

  def __call__(self, batch):
    imgsnp = batch['image']
    assert len(imgsnp.shape) == 4, imgsnp.shape
    assert imgsnp.shape[3] == self.input_ch

    imgs = torch.stack([self.img_transform(x) for x in imgsnp])

    imgs = Variable(imgs)
    if torch.cuda.is_available():
      imgs = imgs.cuda()

    feats = self.G(imgs)
    outputs1 = self.C1(feats)
    outputs2 = self.C2(feats)
    outputs_ensemble = outputs1 + outputs2
    preds_ensemble = outputs_ensemble.data.max(1)[1].cpu().numpy()
    preds_ensemble = preds_ensemble * (360. / 12.)
    return {'yaw': preds_ensemble}


