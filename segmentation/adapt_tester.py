import argparse
import json
import os, os.path as op
from pprint import pprint

import numpy as np
import logging
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from argmyparse import add_additional_params_to_args
from argmyparse import fix_img_shape_args
from datasets import get_dataset
from models.model_util import get_models
from transform import Scale
from util import mkdir_if_not_exist, save_dic_to_json, check_if_done

parser = argparse.ArgumentParser(description='Adapt tester for validation data')
parser.add_argument('tgt_dataset', type=str, choices=["gta", "city", "test", "ir", "city16", "citycam", "scotty"])
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
label_transform = Compose([Scale(train_img_shape, Image.BILINEAR), ToTensor()])

tgt_dataset = get_dataset(dataset_name=args.tgt_dataset, split=args.split, img_transform=img_transform,
                          label_transform=label_transform, test=True, input_ch=train_args.input_ch,
                          keys_dict={'image': 'image', 'image_original': 'image_original', 'url': 'url'})
target_loader = data.DataLoader(tgt_dataset, batch_size=1, pin_memory=True, shuffle=False)

try:
    G, F1, F2 = get_models(net_name=train_args.net, res=train_args.res, input_ch=train_args.input_ch,
                           n_class=train_args.n_class,
                           method=train_args.method, is_data_parallel=train_args.is_data_parallel,use_ae=args.use_ae)
except AttributeError:
    G, F1, F2 = get_models(net_name=train_args.net, res=train_args.res, input_ch=train_args.input_ch,
                           n_class=train_args.n_class,
                           method="MCD", is_data_parallel=False)

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
    import sys
    sys.path.insert(0, os.path.join(os.getenv('CITY_PATH'), 'src'))
    from db.lib.dbExport import DatasetWriter
    out_db_file = os.path.abspath(os.path.join(base_outdir, "predicted.db"))
    writer = DatasetWriter(out_db_file=out_db_file, overwrite=True)

for index, batch in tqdm(enumerate(target_loader)):
    imgs, paths = batch['image'], batch['url']
    assert imgs.size()[0] == 1, 'Batch size is supposed to be 1.'
    path = paths[0]

    imgs = Variable(imgs)
    if torch.cuda.is_available():
        imgs = imgs.cuda()

    feature = G(imgs)
    outputs = F1(feature)

    if args.use_f2:
        outputs += F2(feature)

    if args.saves_prob:
        # Save probability tensors
        prob_outdir = os.path.join(base_outdir, "prob")
        mkdir_if_not_exist(prob_outdir)
        prob_outfn = os.path.join(prob_outdir, path.split('/')[-1].replace('png', 'npy'))
        np.save(prob_outfn, outputs[0].data.cpu().numpy())

    # Save predicted pixel labels(pngs)
    if args.add_bg_loss:
        pred = outputs[0, :args.n_class].data.max(0)[1].cpu()
    else:
        pred = outputs[0, :args.n_class - 1].data.max(0)[1].cpu()

    mask = Image.fromarray(np.uint8(pred.numpy()))
    mask = mask.resize(test_img_shape, Image.NEAREST)
    if args.tgt_dataset == 'citycam':
        imagenp = batch['image_original'][0].numpy()
        masknp = np.array(mask) < 0.5  # Background was 1.
        writer.add_image(imagenp, mask=masknp)
    else:
        label_outdir = os.path.join(base_outdir, "label")
        if index == 0:
            print ("pred label dir: %s" % label_outdir)
        mkdir_if_not_exist(label_outdir)
        label_fn = os.path.join(label_outdir, '%s.png' % op.splitext(op.basename(path))[0])
        mask.save(label_fn)

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
        vis_fn = os.path.join(vis_outdir, path.split('/')[-1]) + '.png'
        mask.save(vis_fn)

if args.tgt_dataset == 'citycam':
    writer.close()
