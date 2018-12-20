import argparse
import json
import os
from pprint import pprint
import progressbar
import numpy as np
import logging
import cv2

import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor

from argmyparse import add_additional_params_to_args
from argmyparse import fix_img_shape_args
from datasets import get_dataset
from models.model_util import get_models
from transform import Scale
from util import mkdir_if_not_exist, save_dic_to_json, check_if_done

parser = argparse.ArgumentParser(description='Adapt tester for validation data')
parser.add_argument('tgt_dataset', type=str, choices=["gta", "city", "test", "ir", "city16", "citycam"])
parser.add_argument('trained_checkpoint', type=str, metavar="PTH.TAR")
parser.add_argument('--outdir', type=str, default="test_output",
                    help='output directory')
parser.add_argument('--test_img_shape', default=(2048, 1024), nargs=2,
                    help="W H, FOR Valid(2048, 1024) Test(1280, 720)")
parser.add_argument('--net', type=str, default="fcn",
                    help="choose from ['fcn','fcnvgg', 'psp', 'segnet','drn_d_105']")
parser.add_argument('--res', type=str, default='50',
                    help='which resnet 18,50,101,152')
parser.add_argument("--input_ch", type=int, default=3,
                    choices=[1, 3, 4])
parser.add_argument('--uses_one_classifier', action="store_true",
                    help="separate f1, f2")
parser.add_argument('--split', type=str, default='val', help="'val' or 'test')  is used")
parser.add_argument("--add_bg_loss", action="store_true",
                    help='whether you add background loss or not')
parser.add_argument("--saves_prob", action="store_true",
                    help='whether you save probability tensors')
parser.add_argument("--use_f2", action="store_true",
                    help='whether you use f2')
parser.add_argument('--use_ae', action="store_true",
                    help="use ae or not")
parser.add_argument('--logging', type=int, choices=[10,20,30,40], default=20)

args = parser.parse_args()
args = add_additional_params_to_args(args)
args = fix_img_shape_args(args)

logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')

indir, infn = os.path.split(args.trained_checkpoint)

trained_mode = indir.split(os.path.sep)[-2]
args.mode = "%s---%s-%s" % (trained_mode, args.tgt_dataset, args.split)

model_name = infn.replace(".pth", "")
if args.use_f2:
    model_name += "-use_f2"

print("=> loading checkpoint '{}'".format(args.trained_checkpoint))
if not os.path.exists(args.trained_checkpoint):
    raise OSError("%s does not exist!" % args.trained_checkpoint)

checkpoint = torch.load(args.trained_checkpoint)
train_args = checkpoint["args"]
args.start_epoch = checkpoint['epoch']
print ("----- train args ------")
pprint(checkpoint["args"].__dict__, indent=4)
print ("-" * 50)
args.train_img_shape = checkpoint["args"].train_img_shape
print("=> loaded checkpoint '{}'".format(args.trained_checkpoint))

base_outdir = os.path.join(args.outdir, args.mode, model_name)
mkdir_if_not_exist(base_outdir)

json_fn = os.path.join(base_outdir, "param.json")
check_if_done(json_fn)
args.machine = os.uname()[1]
save_dic_to_json(args.__dict__, json_fn)

train_img_shape = tuple([int(x) for x in train_args.train_img_shape])
test_img_shape = tuple([int(x) for x in args.test_img_shape])

img_transform = Compose([
    Scale(train_img_shape, Image.BILINEAR),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),

])

tgt_dataset = get_dataset(dataset_name=args.tgt_dataset, split=args.split, img_transform=img_transform,
                          label_transform=None, test=True, input_ch=train_args.input_ch,
                          keys_dict={'image': 'image', 'image_original': 'image_original', 'url': 'url'})
target_loader = data.DataLoader(tgt_dataset, batch_size=10, pin_memory=True, shuffle=False)

G, F1, F2 = get_models(
    net_name=train_args.net, res=train_args.res, input_ch=train_args.input_ch, n_class=train_args.n_class,
    method=train_args.method, 
    is_data_parallel=train_args.is_data_parallel, yaw_loss=train_args.yaw_loss)

G.load_state_dict(checkpoint['g_state_dict'])
F1.load_state_dict(checkpoint['f1_state_dict'])

if args.use_f2:
    F2.load_state_dict(checkpoint['f2_state_dict'])
print("=> loaded checkpoint '{}' (epoch {})"
      .format(args.trained_checkpoint, checkpoint['epoch']))

G.eval()
F1.eval()
F2.eval()

if torch.cuda.is_available():
    G.cuda()
    F1.cuda()
    F2.cuda()

if args.tgt_dataset == 'citycam':
    import os, sys
    sys.path.insert(0, os.path.join(os.getenv('HOME'), 'projects/shuffler/lib'))
    from interfaceWriter import DatasetVideoWriter
    out_db_file = os.path.abspath(os.path.join(base_outdir, "predicted.db"))
    writer_prob = DatasetVideoWriter(out_db_file=out_db_file, rootdir=os.getenv('CITY_PATH'), overwrite=True)
    out_db_file = os.path.abspath(os.path.join(base_outdir, "predictedtop.db"))
    writer_top = DatasetVideoWriter(out_db_file=out_db_file, rootdir=os.getenv('CITY_PATH'), overwrite=True)

widgets = [ progressbar.Counter('Batch: %(value)d/%(max_value)d') ]
bar = progressbar.ProgressBar(max_value=len(target_loader), widgets=widgets, redirect_stdout=True)

for index, batch in bar(enumerate(target_loader)):
    assert 'image' in batch and 'url' in batch, batch.keys()
    imgs, paths = batch['image'], batch['url']
    imgs = Variable(imgs)
    if torch.cuda.is_available():
        imgs = imgs.cuda()

    feature = G(imgs)
    pred_masks, pred_yaws = F1(feature)
    pred_yaws = zip(*pred_yaws)

    # Save predicted pixel labels(pngs)
    for path, pred_mask, pred_yaw in zip(paths, pred_masks, pred_yaws):
        logging.debug('Working on item "%s"' % path)

        if train_args.add_bg_loss:
            pred_mask = pred_mask[:train_args.n_class].data.cpu()
        else:
            pred_yaw = pred_yaw[:train_args.n_class - 1].data.cpu()

        assert args.tgt_dataset == 'citycam'
        pred_mask = torch.softmax(pred_mask, dim=0)
        # Write the probability.
        prob = pred_mask[0]  # Take the first channel, which is the object.
        mask = np.uint8((prob * 255).numpy())
        mask = cv2.resize(mask, dsize=test_img_shape, interpolation=cv2.INTER_NEAREST)
        writer_prob.addImage(mask=mask, imagefile=path, width=test_img_shape[0], height=test_img_shape[1])
        # Write the argmax class
        argmax_pred_mask = np.argmax(pred_mask.numpy(), axis=0)
        mask = 255 - np.uint8(argmax_pred_mask * 255)
        mask = cv2.resize(mask, dsize=test_img_shape, interpolation=cv2.INTER_NEAREST)
        writer_top.addImage(mask=mask, imagefile=path)


if args.tgt_dataset == 'citycam':
    writer_prob.close()
    writer_top.close()
