import argparse
import json
import os
import logging
from pprint import pprint
import cv2
import progressbar
import numpy as np

from PIL import Image
import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor

from argmyparse import add_additional_params_to_args, fix_img_shape_args
from datasets import get_dataset
from models.model_util import get_full_model, get_optimizer
from transform import Scale, ReLabel, ToLabel
from util import mkdir_if_not_exist, save_dic_to_json, check_if_done
from loss import get_yaw_loss  # To process yaw.

parser = argparse.ArgumentParser(description='Adapt tester for validation data')
parser.add_argument('tgt_dataset', type=str, choices=["gta", "city", "test", "ir", "city16", "synthia", "2d3d", 'citycam'])
parser.add_argument('--split', type=str, default='val', help="'val' or 'test')  is used")
parser.add_argument('trained_checkpoint', type=str, metavar="PTH")
parser.add_argument('--outdir', type=str, default="test_output",
                    help='output directory')
parser.add_argument('--test_img_shape', default=(2048, 1024), nargs=2,
                    help="W H, FOR Valid(2048, 1024) Test(1280, 720)")
parser.add_argument("---saves_prob", action="store_true",
                    help='whether you save probability tensors')
parser.add_argument('--logging', type=int, choices=[10,20,30,40], default=20)

args = parser.parse_args()
args = add_additional_params_to_args(args)
args = fix_img_shape_args(args)

FORMAT = '[%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s'
logging.basicConfig(level=args.logging, format=FORMAT)

if not os.path.exists(args.trained_checkpoint):
    raise OSError("%s does not exist!" % args.trained_checkpoint)

checkpoint = torch.load(args.trained_checkpoint)
train_args = checkpoint['args']  # Load args!
model = get_full_model(train_args.net, train_args.n_class, train_args.input_ch,
    yaw_loss=train_args.yaw_loss)
model.load_state_dict(checkpoint['state_dict'])
model.eval()
if torch.cuda.is_available():
    model.cuda()

print ("----- train args ------")
pprint(checkpoint["args"].__dict__, indent=4)
print ("-" * 50)
args.train_img_shape = checkpoint["args"].train_img_shape
print("=> loaded checkpoint '{}'".format(args.trained_checkpoint))

indir, infn = os.path.split(args.trained_checkpoint)

trained_mode = indir.split(os.path.sep)[-2]
args.mode = "%s---%s-%s" % (trained_mode, args.tgt_dataset, args.split)
model_name = infn.replace(".pth", "")

base_outdir = os.path.join(args.outdir, args.mode, model_name)
mkdir_if_not_exist(base_outdir)

json_fn = os.path.join(base_outdir, "param.json")
check_if_done(json_fn)
args.machine = os.uname()[1]
save_dic_to_json(args.__dict__, json_fn)

train_img_shape = tuple([int(x) for x in args.train_img_shape])
test_img_shape = tuple([int(x) for x in args.test_img_shape])

img_transform = Compose([
    Scale(train_img_shape, Image.BILINEAR),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),

])
label_transform = Compose([
    Scale(train_img_shape, Image.NEAREST),
    ToLabel(),
    ReLabel(255, train_args.n_class - 1),
])

tgt_dataset = get_dataset(dataset_name=args.tgt_dataset, split=args.split, img_transform=img_transform,
                          label_transform=label_transform, test=True, input_ch=train_args.input_ch,
                          keys_dict={'image': 'image', 'image_original': 'image_original',
                                     'url': 'url', 'objectid': 'objectid'})

target_loader = data.DataLoader(tgt_dataset, batch_size=10, pin_memory=True)

# Use it to extract the value of yaw from the prediction.
criterion_yaw = get_yaw_loss(checkpoint["args"].yaw_loss)
criterion_yaw.eval()
if torch.cuda.is_available():
    criterion_yaw = criterion_yaw.cuda()

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

for ind, batch in bar(enumerate(target_loader)):
    assert 'image' in batch and 'url' in batch, batch.keys()
    imgs, paths, objectids = batch['image'], batch['url'], batch['objectid'].numpy().tolist()
    if torch.cuda.is_available():
        imgs = imgs.cuda()

    preds_mask, preds_yaw = model(imgs)

    # Extract the mask.
    preds_mask = torch.softmax(preds_mask, dim=1)
    if train_args.add_bg_loss:
        preds_mask = preds_mask[:,:train_args.n_class]
    else:
        preds_mask = preds_mask[:,:train_args.n_class-1]
    preds_mask = preds_mask.data.cpu().numpy()

    # Extract the yaw.
    preds_yaw = criterion_yaw.prediction2angle(preds_yaw)
    preds_yaw = preds_yaw.data.cpu().numpy()

    for path, objectid, pred_mask, pred_yaw in zip(paths, objectids, preds_mask, preds_yaw):
        logging.debug('Working on item "%s"' % path)

        # Write the probability.
        prob = pred_mask[0]  # Take the first channel, which is the object.
        prob = np.uint8(prob * 255)
        prob = cv2.resize(prob, dsize=test_img_shape, interpolation=cv2.INTER_NEAREST)
        writer_prob.addImage(mask=prob, imagefile=path, width=test_img_shape[0], height=test_img_shape[1])
        writer_prob.addObject({'objectid': objectid, 'yaw': pred_yaw, 'imagefile': path})
        # Write the argmax class
        argmax_mask = np.argmax(pred_mask, axis=0)
        pred_mask = 255 - np.uint8(argmax_mask * 255)
        pred_mask = cv2.resize(pred_mask, dsize=test_img_shape, interpolation=cv2.INTER_NEAREST)
        writer_top.addImage(mask=pred_mask, imagefile=path)
        writer_top.addObject({'objectid': objectid, 'yaw': pred_yaw, 'imagefile': path})

writer_prob.close()
writer_top.close()
