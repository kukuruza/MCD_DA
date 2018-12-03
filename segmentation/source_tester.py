import argparse
import json
import os
import logging
from pprint import pprint
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm
import cv2

from argmyparse import add_additional_params_to_args, fix_img_shape_args
from datasets import get_dataset
from models.model_util import get_full_model, get_optimizer
from transform import Scale, ReLabel, ToLabel
from util import mkdir_if_not_exist, save_dic_to_json, check_if_done

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
model = get_full_model(train_args.net, train_args.res, train_args.n_class, train_args.input_ch)
model.load_state_dict(checkpoint['state_dict'])
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
                          keys_dict={'image': 'image', 'image_original': 'image_original', 'url': 'url'})

target_loader = data.DataLoader(tgt_dataset, batch_size=10, pin_memory=True)

if torch.cuda.is_available():
    model.cuda()

model.eval()

if args.tgt_dataset == 'citycam':
    import os, sys
    sys.path.insert(0, os.path.join(os.getenv('HOME'), 'projects/shuffler/lib'))
    from interfaceWriter import DatasetVideoWriter
    out_db_file = os.path.abspath(os.path.join(base_outdir, "predicted.db"))
    writer_prob = DatasetVideoWriter(out_db_file=out_db_file, rootdir=os.getenv('CITY_PATH'), overwrite=True)
    out_db_file = os.path.abspath(os.path.join(base_outdir, "predictedtop.db"))
    writer_top = DatasetVideoWriter(out_db_file=out_db_file, rootdir=os.getenv('CITY_PATH'), overwrite=True)

for index, batch in tqdm(enumerate(target_loader)):
    assert 'image' in batch and 'url' in batch, batch.keys()
    imgs, paths = batch['image'], batch['url']
    imgs = Variable(imgs)
    if torch.cuda.is_available():
        imgs = imgs.cuda()

    preds, yaws = model(imgs)

    # Save predicted pixel labels(pngs)
    for path, pred in zip(paths, preds):
        logging.debug('Working on item "%s"' % path)

        if train_args.add_bg_loss:
            pred = pred[:train_args.n_class].data.cpu()
        else:
            pred = pred[:train_args.n_class - 1].data.cpu()

        if args.tgt_dataset == 'citycam':
            # Write the probability.
            prob = pred[0]  # Take the first channel, which is the object.
            mask = np.uint8((prob * 255).numpy())
            mask = cv2.resize(mask, dsize=test_img_shape, interpolation=cv2.INTER_NEAREST)
            writer_prob.addImage(mask=mask, imagefile=path, width=test_img_shape[0], height=test_img_shape[1])
            # Write the argmax class
            argmax_pred = np.argmax(pred.numpy(), axis=0)
            mask = 255 - np.uint8(argmax_pred * 255)
            mask = cv2.resize(mask, dsize=test_img_shape, interpolation=cv2.INTER_NEAREST)
            writer_top.addImage(mask=mask, imagefile=path)
        else:
            argmax_pred = pred.max(0)[1]
            mask = Image.fromarray(np.uint8(pred.numpy()))
            mask = mask.resize(test_img_shape, Image.NEAREST)
            label_outdir = os.path.join(base_outdir, "label")
            if index == 0:
                print ("pred label dir: %s" % label_outdir)
            mkdir_if_not_exist(label_outdir)
            label_fn = os.path.join(label_outdir, path.split('/')[-1])
            mask.save(label_fn)

            #  Save visualized predicted pixel labels(pngs)
            if args.tgt_dataset in ["city16", "synthia"]:
                info_json_fn = "./dataset/synthia2cityscapes_info.json"
            else:
                info_json_fn = "./dataset/city_info.json"

            # Save visualized predicted pixel labels(pngs)
            with open(info_json_fn) as f:
                city_info_dic = json.load(f)

            palette = np.array(city_info_dic['palette'], dtype=np.uint8)
            mask.putpalette(palette.flatten())
            vis_outdir = os.path.join(base_outdir, "vis")
            mkdir_if_not_exist(vis_outdir)
            vis_fn = os.path.join(vis_outdir, path.split('/')[-1])
            mask.save(vis_fn)

if args.tgt_dataset == 'citycam':
    writer_prob.close()
    writer_top.close()
