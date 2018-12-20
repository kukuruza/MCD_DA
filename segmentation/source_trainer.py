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
from tensorboard_logger import configure as tfconfigure, log_value, log_histogram

from argmyparse import get_src_only_training_parser, add_additional_params_to_args, fix_img_shape_args
from datasets import get_dataset
from loss import get_yaw_loss
from models.model_util import get_optimizer, get_full_model
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

src_dataset = get_dataset(dataset_name=args.src_dataset, split=args.split, img_transform=img_transform,
                          label_transform=label_transform, test=False, input_ch=args.input_ch,
                          keys_dict={'image': 'image', 'image_original': 'image_original', 
                              'mask': 'mask', 'yaw': 'yaw', 'url': 'url'})

kwargs = {'num_workers': 5, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = torch.utils.data.DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

model = get_full_model(net=args.net, res=args.res, n_class=args.n_class, 
    input_ch=args.input_ch, yaw_loss=args.yaw_loss)
optimizer = get_optimizer(model.parameters(), opt=args.opt, lr=args.lr,
    momentum=args.momentum, weight_decay=args.weight_decay)

if args.resume:
    print("=> loading checkpoint '{}'".format(args.resume))
    if not os.path.exists(args.resume):
        raise OSError("%s does not exist!" % args.resume)

    checkpoint = torch.load(args.resume)

    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if torch.cuda.is_available():
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
    model.cuda()
    weight = weight.cuda()
model.train()

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

        imgs, gt_masks, gt_yaws = batch['image'], batch['mask'], batch['yaw'].float()
        imgs, gt_masks, gt_yaws = Variable(imgs), Variable(gt_masks), Variable(gt_yaws)
        if torch.cuda.is_available():
            imgs, gt_masks, gt_yaws = imgs.cuda(), gt_masks.cuda(), gt_yaws.cuda()

        # update generator and classifiers by source samples
        optimizer.zero_grad()
        pred_masks, pred_yaws = model(imgs)
        
        loss_mask = criterion_mask(pred_masks, gt_masks)
        loss_yaw = criterion_yaw(pred_yaws, gt_yaws)
        metrics_yaw = criterion_yaw.metrics(pred_yaws, gt_yaws)

        loss = loss_mask + loss_yaw
        loss.backward()
        tflogger.acc_value('train/loss/total', loss / args.batch_size)
        tflogger.acc_value('train/loss/mask', loss_mask / args.batch_size)
        tflogger.acc_value('train/loss/yaw', loss_yaw / args.batch_size)
        tflogger.acc_value('train/metrics/yaw', metrics_yaw.mean())
        tflogger.acc_histogram('train/hist/yaw', metrics_yaw)

        optimizer.step()

        if (log_counter + 1) % args.freq_log == 0:
            print 'Epoch %d/%d, batch %d/%d' % \
                (epoch, args.epochs, ibatch, len(train_loader)), tflogger.get_mean_values()
            tflogger.flush (step=log_counter-1)

        if args.max_iter is not None and ibatch >= args.max_iter:
            break

    log_value('lr', args.lr, epoch)
    if args.adjust_lr:
        args.lr = adjust_learning_rate(optimizer, args.lr, args.weight_decay, epoch, args.epochs)

    if args.net == "fcn" or args.net == "psp":
        checkpoint_fn = os.path.join(args.pth_dir, "%s-%s-res%s-%s.pth.tar" % (
            args.savename, args.net, args.res, epoch + 1))
    else:
        checkpoint_fn = os.path.join(args.pth_dir, "%s-%s-%s.pth.tar" % (
            args.savename, args.net, epoch + 1))

    save_dic = {
        'args': args,
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    save_checkpoint(save_dic, is_best=False, filename=checkpoint_fn)
