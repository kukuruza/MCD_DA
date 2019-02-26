from __future__ import division

import os
import logging
from pprint import pprint
import progressbar
import numpy as np

from PIL import Image
import torch
from torch.autograd import Variable
from torch.utils import data
import torch.nn
from torchvision.transforms import Compose, Normalize, ToTensor
torch.set_printoptions(linewidth=150, precision=2)
torch.set_flush_denormal(True)
from tensorboard_logger import configure as tfconfigure, log_value, log_histogram, log_images

from argmyparse import get_src_only_training_parser, add_additional_params_to_args, fix_img_shape_args
from datasets import get_dataset
from loss import get_yaw_loss
from models.model_util import get_models, get_optimizer
from transform import ReLabel, ToLabel, Scale, RandomSizedCrop, RandomHorizontalFlip, RandomRotation
from util import check_if_done, save_checkpoint, adjust_learning_rate, emphasize_str, get_class_weight_from_file
from util import mkdir_if_not_exist, save_dic_to_json
from util import AccumulatedTFLogger

parser = get_src_only_training_parser()
parser.add_argument('--logging', type=int, choices=[10,20,30,40], default=20)
args = parser.parse_args()
args = add_additional_params_to_args(args)
args = fix_img_shape_args(args)

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
    ReLabel(255, args.n_class - 1),
])

keys_dict={'image': 'image', 'image_original': 'image_original', 'mask': 'mask', 'url': 'url'}
if args.weight_yaw > 0:
    keys_dict['yaw'] = 'yaw'
src_dataset = get_dataset(dataset_name=args.src_dataset, split=args.split, img_transform=img_transform,
                          label_transform=label_transform, test=False, input_ch=args.input_ch, keys_dict=keys_dict)

kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = torch.utils.data.DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

model_g, model_f1, model_f2 = get_models(
    net_name=args.net, res=args.res, input_ch=args.input_ch, n_class=args.n_class,
    is_data_parallel=args.is_data_parallel, yaw_loss=args.yaw_loss)

optimizer_g = get_optimizer(model_g.parameters(),
    lr=args.lr, momentum=args.momentum, opt=args.opt, weight_decay=args.weight_decay)
optimizer_f = get_optimizer(list(model_f1.parameters()) + list(model_f2.parameters()),
    lr=args.lr, momentum=args.momentum, opt=args.opt, weight_decay=args.weight_decay)

if args.resume:
    print("=> loading checkpoint '{}'".format(args.resume))
    if not os.path.exists(args.resume):
        raise OSError("%s does not exist!" % args.resume)

    checkpoint = torch.load(args.resume)

    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_f1.load_state_dict(checkpoint['f1_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g'])
    optimizer_f.load_state_dict(checkpoint['optimizer_f'])
    if torch.cuda.is_available():
        for optimizer in [optimizer_g, optimizer_f]:  # https://github.com/pytorch/pytorch/issues/2830
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
    print("=> loaded checkpoint '{}'".format(args.resume))
    start_epoch = checkpoint["epoch"]
    log_counter = checkpoint['log_counter'] if 'log_counter' in checkpoint else start_epoch * len(train_loader)
else:
    start_epoch = 0
    log_counter = 0

if args.net in ["fcn", "psp"]:
    model_name = "%s-%s-res%s" % (args.savename, args.net, args.res)
else:
    model_name = "%s-%s" % (args.savename, args.net)

mode =  "%s-%s_only_%sch" % (args.src_dataset, args.split, args.input_ch)
outdir = os.path.join(args.base_outdir, mode)

# Create Model Dir
args.pth_dir = os.path.join(outdir, "pth")
mkdir_if_not_exist(args.pth_dir)

# Set TF-Logger
args.tflog_dir = os.path.join(outdir, "tflog", model_name)
mkdir_if_not_exist(args.tflog_dir)
tfconfigure(args.tflog_dir, flush_secs=10)
tflogger = AccumulatedTFLogger()

# Save param dic
if args.resume:
    json_fn = os.path.join(outdir, "param_%s_resume.json" % args.savename)
else:
    json_fn = os.path.join(outdir, "param-%s.json" % model_name)
check_if_done(json_fn)
args.machine = os.uname()[1]
save_dic_to_json(args.__dict__, json_fn)

weight = get_class_weight_from_file(n_class=args.n_class, 
    weight_filename=args.loss_weights_file, add_bg_loss=args.add_bg_loss)

if torch.cuda.is_available():
    model_g.cuda()
    model_f1.cuda()
    weight = weight.cuda()

model_g.train()
model_f1.train()

criterion_mask = torch.nn.CrossEntropyLoss(weight=weight)
criterion_yaw = get_yaw_loss(args.yaw_loss)
if torch.cuda.is_available():
    criterion_mask = criterion_mask.cuda()
    criterion_yaw = criterion_yaw.cuda()

for epoch in range(start_epoch, args.epochs):
    widgets = [
        'Epoch %d/%d,' % (epoch, args.epochs),
        ' ', progressbar.Counter('batch %(value)d/%(max_value)d')
    ]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=len(train_loader), redirect_stdout=True)

    for ibatch, batch in bar(enumerate(train_loader)):
        log_counter += 1

        imgs, gt_masks = batch['image'], batch['mask']
        imgs, gt_masks = Variable(imgs), Variable(gt_masks)
        if torch.cuda.is_available():
            imgs, gt_masks = imgs.cuda(), gt_masks.cuda()

        optimizer_f.zero_grad()
        optimizer_g.zero_grad()

        # update generator and classifiers by source samples
        features = model_g(imgs)
        pred_masks, _ = model_f1(features)

        loss_mask = criterion_mask(pred_masks, gt_masks)

        loss = loss_mask
        loss.backward()
        tflogger.acc_value('train/loss/total', loss / args.batch_size)
        tflogger.acc_value('train/loss/mask', loss_mask / args.batch_size)

        optimizer_g.step()
        optimizer_f.step()

        if (log_counter + 1) % args.freq_log == 0:
            print 'Epoch %d/%d, batch %d/%d' % \
                (epoch, args.epochs, ibatch, len(train_loader)), tflogger.get_mean_values()
            log_images('train/gt/imgs', imgs[:1].cpu().data, step=log_counter)
            log_images('train/gt/masks', gt_masks[:1].cpu().data, step=log_counter)
            pred_masks_asimage = torch.softmax(pred_masks, dim=1)[:1,1].cpu().data
            #print ('pred_masks_asimage', pred_masks_asimage.min(), pred_masks_asimage.max(), pred_masks_asimage.dtype)
            log_images('train/pred/masks', pred_masks_asimage, step=log_counter)
            tflogger.flush (step=log_counter-1)

        if args.max_iter is not None and ibatch >= args.max_iter:
            break

    log_value('lr', args.lr, epoch)
    if args.adjust_lr:
        args.lr = adjust_learning_rate(optimizer_g, args.lr, args.weight_decay, epoch, args.epochs)
        args.lr = adjust_learning_rate(optimizer_f, args.lr, args.weight_decay, epoch, args.epochs)

    if args.net == "fcn" or args.net == "psp":
        checkpoint_fn = os.path.join(args.pth_dir, "%s-%s-res%s-%s.pth.tar" % (
            args.savename, args.net, args.res, epoch + 1))
    else:
        checkpoint_fn = os.path.join(args.pth_dir, "%s-%s-%s.pth.tar" % (
            args.savename, args.net, epoch + 1))

    save_dic = {
        'epoch': epoch + 1,
        'args': args,
        'g_state_dict': model_g.state_dict(),
        'f1_state_dict': model_f1.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_f': optimizer_f.state_dict(),
    }

    if (epoch+1) % args.freq_checkpoint == 0:
        save_checkpoint(save_dic, is_best=False, filename=checkpoint_fn)
