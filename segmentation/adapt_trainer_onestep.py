from __future__ import division

import os
import progressbar
import numpy as np
from pprint import pprint

from PIL import Image
import torch
from torch.autograd import Variable
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor
from transform import ReLabel, ToLabel, Scale, RandomSizedCrop, RandomHorizontalFlip, RandomRotation
from tensorboard_logger import configure as tfconfigure, log_value, log_histogram, log_images

from argmyparse import add_additional_params_to_args, fix_img_shape_args, get_da_mcd_training_parser
from datasets import ConcatDataset, get_dataset, check_src_tgt_ok
from loss import get_yaw_loss, get_prob_distance_criterion
from models.model_util import get_models, get_optimizer
from util import mkdir_if_not_exist, save_dic_to_json, check_if_done, save_checkpoint
from util import adjust_learning_rate, get_class_weight_from_file
from util import AccumulatedTFLogger

# from visualize import LinePlotter
#set_debugger_org_frc()
parser = get_da_mcd_training_parser()
args = parser.parse_args()
args = add_additional_params_to_args(args)
args = fix_img_shape_args(args)
check_src_tgt_ok(args.src_dataset, args.tgt_dataset)

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
    ReLabel(255, args.n_class - 1),  # Last Class is "Void" or "Background" class
])

src_dataset = get_dataset(dataset_name=args.src_dataset, split=args.src_split, img_transform=img_transform,
                          label_transform=label_transform, test=False, input_ch=args.input_ch,
                          keys_dict={'image': 'S_image', 'mask': 'S_mask', 'yaw': 'S_yaw'})

if args.num_of_labelled_target > 0:
  tgt_dataset = get_dataset(dataset_name=args.tgt_dataset, split=args.tgt_split, img_transform=img_transform,
                          label_transform=label_transform, test=False, input_ch=args.input_ch,
                          keys_dict={'image': 'T_image', 'mask': 'T_mask', 'index': 'T_index'})
else:
  tgt_dataset = get_dataset(dataset_name=args.tgt_dataset, split=args.tgt_split, img_transform=img_transform,
                          label_transform=label_transform, test=False, input_ch=args.input_ch,
                          keys_dict={'image': 'T_image'})

concat_dataset = ConcatDataset([src_dataset, tgt_dataset])
train_loader = torch.utils.data.DataLoader(
    concat_dataset,
    batch_size=args.batch_size, shuffle=True,
    pin_memory=True)

model_g, model_f1, model_f2 = get_models(
    net_name=args.net, res=args.res, input_ch=args.input_ch, n_class=args.n_class,
    is_data_parallel=args.is_data_parallel, yaw_loss=args.yaw_loss)

optimizer_g = get_optimizer(model_g.parameters(),
    lr=args.lr, momentum=args.momentum, opt=args.opt, weight_decay=args.weight_decay)
optimizer_f = get_optimizer(list(model_f1.parameters()) + list(model_f2.parameters()),
    lr=args.lr, momentum=args.momentum, opt=args.opt, weight_decay=args.weight_decay)

if args.resume:
    print('Loading checkpoint %s' % args.resume)
    if not os.path.exists(args.resume):
        raise OSError("%s does not exist!" % args.resume)

    checkpoint = torch.load(args.resume)

    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_f1.load_state_dict(checkpoint['f1_state_dict'])
    if not args.uses_one_classifier:
        model_f2.load_state_dict(checkpoint['f2_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g'])
    optimizer_f.load_state_dict(checkpoint['optimizer_f'])
    if torch.cuda.is_available():
        for optimizer in [optimizer_g, optimizer_f]:  # https://github.com/pytorch/pytorch/issues/2830
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.cuda()
    print('Loaded checkpoint.')
    start_epoch = checkpoint["epoch"]
    log_counter = checkpoint['log_counter'] if 'log_counter' in checkpoint else start_epoch * len(train_loader)
else:
    start_epoch = 0
    log_counter = 0

if args.net in ["fcn", "psp"]:
    model_name = "%s-%s-%s-res%s" % (args.method, args.savename, args.net, args.res)
else:
    model_name = "%s-%s-%s" % (args.method, args.savename, args.net)

mode = "%s-%s2%s-%s_%sch" % (args.src_dataset, args.src_split, args.tgt_dataset, args.tgt_split, args.input_ch)
outdir = os.path.join(args.base_outdir, mode)

# Create Model Dir
pth_dir = os.path.join(outdir, "pth")
mkdir_if_not_exist(pth_dir)

# Set TF-Logger
tflog_dir = os.path.join(outdir, "tflog", model_name)
mkdir_if_not_exist(tflog_dir)
tfconfigure(tflog_dir, flush_secs=5)
tflogger = AccumulatedTFLogger()

# Save param dic
if args.resume:
    json_fn = os.path.join(outdir, "param-%s_resume.json" % model_name)
else:
    json_fn = os.path.join(outdir, "param-%s.json" % model_name)
check_if_done(json_fn)
save_dic_to_json(args.__dict__, json_fn)

weight = get_class_weight_from_file(n_class=args.n_class,
    weight_filename=args.loss_weights_file, add_bg_loss=args.add_bg_loss)

if torch.cuda.is_available():
    model_g.cuda()
    model_f1.cuda()
    model_f2.cuda()
    weight = weight.cuda()

criterion_mask = torch.nn.CrossEntropyLoss(weight=weight)
criterion_yaw = get_yaw_loss(args.yaw_loss)
criterion_d = get_prob_distance_criterion(args.d_loss)
if torch.cuda.is_available():
    criterion_mask = criterion_mask.cuda()
    criterion_yaw = criterion_yaw.cuda()
    criterion_d = criterion_d.cuda()

model_g.train()
model_f1.train()
model_f2.train()

print ('Will train from epoch %d to epoch %d (exclusively)' % (start_epoch, args.epochs))
for epoch in range(start_epoch, args.epochs):
    widgets = [
        'Epoch %d/%d,' % (epoch, args.epochs),
        ' ', progressbar.Counter('batch %(value)d/%(max_value)d')
    ]
    bar = progressbar.ProgressBar(widgets=widgets, max_value=len(train_loader), redirect_stdout=True)

    for ibatch, batch in bar(enumerate(train_loader)):

        src_imgs, src_masks, tgt_imgs = batch['S_image'], batch['S_mask'], batch['T_image']
        src_imgs, src_masks, tgt_imgs = Variable(src_imgs), Variable(src_masks), Variable(tgt_imgs)
        if args.num_of_labelled_target > 0:
            tgt_masks, tgt_use_masks = batch['T_mask'], batch['T_index'] < args.num_of_labelled_target
            tgt_masks, tgt_use_masks = Variable(tgt_masks), Variable(tgt_use_masks)

        if torch.cuda.is_available():
            src_imgs, src_masks, tgt_imgs = src_imgs.cuda(), src_masks.cuda(), tgt_imgs.cuda()
            if args.num_of_labelled_target > 0:
                tgt_masks, tgt_use_masks = tgt_masks.cuda(), tgt_use_masks.cuda()

        optimizer_g.zero_grad()
        optimizer_f.zero_grad()

        # update generator and classifiers by source samples
        features = model_g(src_imgs)
        pred_src_masks1, _ = model_f1(features)
        pred_src_masks2, _ = model_f2(features)

        c_loss_mask  = criterion_mask(pred_src_masks1, src_masks)
        c_loss_mask += criterion_mask(pred_src_masks2, src_masks)
        c_loss_mask /= 2.
        c_loss = c_loss_mask
        c_loss.backward()
#        c_loss.backward(retain_graph=True)
        tflogger.acc_value('train/loss/src', c_loss / args.batch_size)
        tflogger.acc_value('train/loss/mask', c_loss_mask / args.batch_size)

        # d tgt loss
        lambd = 1.0
        model_f1.set_lambda(lambd)
        model_f2.set_lambda(lambd)
        features = model_g(tgt_imgs)
        pred_tgt_masks1, _ = model_f1(features, reverse=True)
        pred_tgt_masks2, _ = model_f2(features, reverse=True)
        d_loss_mask = - criterion_d(pred_tgt_masks1, pred_tgt_masks2)
        d_loss = d_loss_mask
        if args.num_of_labelled_target > 0 and tgt_use_masks.sum() > 0:
          d_loss.backward(retain_graph=True)
        else:
          d_loss.backward()
        tflogger.acc_value('train/loss/discr_mask', d_loss_mask / args.batch_size)
        tflogger.acc_value('train/loss/discr', d_loss / args.batch_size)

        # c tgt loss
        if args.num_of_labelled_target > 0 and tgt_use_masks.sum() > 0:
            pred_tgt_masks1, _ = model_f1(features)
            pred_tgt_masks2, _ = model_f2(features)
            c_loss_mask  = criterion_mask(pred_tgt_masks1[tgt_use_masks,:,:,:], tgt_masks[tgt_use_masks,:,:])
            c_loss_mask += criterion_mask(pred_tgt_masks2[tgt_use_masks,:,:,:], tgt_masks[tgt_use_masks,:,:])
            c_loss_mask /= 2.
            c_loss_mask[torch.isnan(c_loss_mask)] = 0
            c_loss = c_loss_mask
            c_loss.backward()
            tflogger.acc_value('train/loss/mask_tgt', c_loss_mask / tgt_use_masks.sum().type(torch.float))
            log_images('train/gt/tgt_masks', tgt_masks[:1].cpu().data, step=log_counter)

        optimizer_f.step()
        optimizer_g.step()

        if (log_counter + 1) % args.freq_log == 0:
            print 'Epoch %d/%d, batch %d/%d' % \
                (epoch, args.epochs, ibatch, len(train_loader)), tflogger.get_mean_values()
            log_images('train/gt/src_imgs', src_imgs[:1].cpu().data, step=log_counter)
            log_images('train/gt/tgt_imgs', tgt_imgs[:1].cpu().data, step=log_counter)
            log_images('train/gt/src_masks', src_masks[:1].cpu().data, step=log_counter)
            log_images('train/pred/src_masks1', torch.softmax(pred_src_masks1, dim=1)[:1,1,:,:].cpu().data, step=log_counter)
#            log_images('train/pred/tgt_masks1', torch.softmax(pred_tgt_masks1, dim=1)[:1,1,:,:].cpu().data, step=log_counter)
            log_images('train/pred/src_masks2', torch.softmax(pred_src_masks2, dim=1)[:1,1,:,:].cpu().data, step=log_counter)
#            log_images('train/pred/tgt_masks2', torch.softmax(pred_tgt_masks2, dim=1)[:1,1,:,:].cpu().data, step=log_counter)
            tflogger.flush (step=log_counter)
        log_counter += 1

        if args.max_iter is not None and ibatch > args.max_iter:
            break

    log_value('lr', args.lr, epoch)
    if args.adjust_lr:
        args.lr = adjust_learning_rate(optimizer_g, args.lr, args.weight_decay, epoch, args.epochs)
        args.lr = adjust_learning_rate(optimizer_f, args.lr, args.weight_decay, epoch, args.epochs)

    checkpoint_fn = os.path.join(pth_dir, "%s-%s.pth.tar" % (model_name, epoch + 1))
    steps_fn = os.path.join(args.pth_dir, "%s.steps.txt" % model_name)
    save_dic = {
        'epoch': epoch + 1,
        'args': args,
        'g_state_dict': model_g.state_dict(),
        'f1_state_dict': model_f1.state_dict(),
        'optimizer_g': optimizer_g.state_dict(),
        'optimizer_f': optimizer_f.state_dict(),
        'log_counter': log_counter,
    }
    def get_step():
        return epoch * len(concat_dataset)
    if saved_step not in vars() or float(get_step()) / saved_step > 1 + args.freq_checkpoint:
        save_checkpoint(save_dic, is_best=False, filename=checkpoint_fn)
        with open(steps_fn, 'a') as f:
            f.write('%d %d\n' % (epoch, get_step()))
        saved_step = get_step()

