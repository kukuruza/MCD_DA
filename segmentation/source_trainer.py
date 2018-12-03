from __future__ import division

import os
import logging
from tqdm import tqdm

import torch
from PIL import Image
from tensorboard_logger import configure, log_value
from torch.autograd import Variable
from torch.utils import data
import torch.nn
from torchvision.transforms import Compose, Normalize, ToTensor
torch.set_printoptions(linewidth=150)

from argmyparse import get_src_only_training_parser, add_additional_params_to_args, fix_img_shape_args
from datasets import get_dataset
from loss import CrossEntropyLoss2d
from models.model_util import get_optimizer, get_full_model  # check_training
from transform import ReLabel, ToLabel, Scale, RandomSizedCrop, RandomHorizontalFlip, RandomRotation
from util import check_if_done, save_checkpoint, adjust_learning_rate, emphasize_str, get_class_weight_from_file
from util import mkdir_if_not_exist, save_dic_to_json

parser = get_src_only_training_parser()
parser.add_argument('--logging', type=int, choices=[10,20,30,40], default=20)
args = parser.parse_args()
args = add_additional_params_to_args(args)
args = fix_img_shape_args(args)

FORMAT = '[%(filename)s:%(lineno)s - %(funcName)s() %(levelname)s]: %(message)s'
logging.basicConfig(level=args.logging, format=FORMAT)

if args.resume:
    print("=> loading checkpoint '{}'".format(args.resume))
    if not os.path.exists(args.resume):
        raise OSError("%s does not exist!" % args.resume)

    indir, infn = os.path.split(args.resume)

    old_savename = args.savename
    args.savename = infn.split("-")[0]
    print ("savename is %s (original savename %s was overwritten)" % (args.savename, old_savename))

    checkpoint = torch.load(args.resume)
    args = checkpoint['args']  # Load args!

    model = get_full_model(net=args.net, res=args.res, n_class=args.n_class, input_ch=args.input_ch)
    optimizer = get_optimizer(model.parameters(), opt=args.opt, lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("=> loaded checkpoint '{}'".format(args.resume))

    json_fn = os.path.join(args.outdir, "param_%s_resume.json" % args.savename)
    check_if_done(json_fn)
    args.machine = os.uname()[1]
    save_dic_to_json(args.__dict__, json_fn)

else:
    model = get_full_model(net=args.net, res=args.res, n_class=args.n_class, input_ch=args.input_ch)
    optimizer = get_optimizer(model.parameters(), opt=args.opt, lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)

    args.outdir = os.path.join(args.base_outdir, "%s-%s_only_%sch" % (args.src_dataset, args.split, args.input_ch))
    args.pth_dir = os.path.join(args.outdir, "pth")

    if args.net in ["fcn", "psp"]:
        model_name = "%s-%s-res%s" % (args.savename, args.net, args.res)
    else:
        model_name = "%s-%s" % (args.savename, args.net)

    args.tflog_dir = os.path.join(args.outdir, "tflog", model_name)
    mkdir_if_not_exist(args.pth_dir)
    mkdir_if_not_exist(args.tflog_dir)

    json_fn = os.path.join(args.outdir, "param-%s.json" % model_name)
    check_if_done(json_fn)
    args.machine = os.uname()[1]
    save_dic_to_json(args.__dict__, json_fn)

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
                              'mask': 'label_map', 'yaw': 'yaw', 'url': 'url', 'yaw_discr': 'yaw_discr'})

kwargs = {'num_workers': 5, 'pin_memory': True} if torch.cuda.is_available() else {}
train_loader = torch.utils.data.DataLoader(src_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)

weight = get_class_weight_from_file(n_class=args.n_class, weight_filename=args.loss_weights_file,
                                    add_bg_loss=args.add_bg_loss)

if torch.cuda.is_available():
    model.cuda()
    weight = weight.cuda()

criterion_map = CrossEntropyLoss2d(weight) # torch.nn.CrossEntropyLoss(weight=weight)
criterion_yaw = torch.nn.CrossEntropyLoss() # MSELoss()
if torch.cuda.is_available():
    criterion_map = criterion_map.cuda()
    criterion_yaw = criterion_yaw.cuda()

configure(args.tflog_dir, flush_secs=5)

model.train()

for epoch in range(args.epochs):
    cum_losses = {'count': 0, 'total': 0, 'mask': 0, 'yaw': 0}
    for ind, batch in tqdm(enumerate(train_loader)):
        imgs, lbls, yaws = batch['image'], batch['label_map'], batch['yaw_discr']

        imgs, lbls, yaws = Variable(imgs), Variable(lbls), Variable(yaws)
        if torch.cuda.is_available():
            imgs, lbls, yaws = imgs.cuda(), lbls.cuda(), yaws.cuda()

        # update generator and classifiers by source samples
        optimizer.zero_grad()
        preds_mask, preds_yaw = model(imgs)

        loss_mask = criterion_map(preds_mask, lbls)
        loss_yaw = criterion_yaw(preds_yaw, yaws)

        loss = loss_mask # + loss_yaw
        loss.backward()
        cum_losses['count'] += 1
        cum_losses['total'] += loss.data
        cum_losses['mask'] += loss_mask.data
        cum_losses['yaw'] += loss_yaw.data

        optimizer.step()

        if ind % 50 == 0:
            count = cum_losses['count']
            print (preds_yaw, yaws)
            print("iter [%d] MapLoss: %.4f, YawLoss: %.4f, Total CLoss: %.4f" %
                    (ind, cum_losses['mask'] / count, cum_losses['yaw'] / count, cum_losses['total'] / count))
            cum_losses = {'count': 0, 'total': 0, 'mask': 0, 'yaw': 0}

        if args.max_iter is not None and ind > args.max_iter:
            break

#    print("Epoch [%d] Loss: %.4f" % (epoch + 1, epoch_loss))
#    log_value('loss', epoch_loss, epoch)
    log_value('lr', args.lr, epoch)

    if args.adjust_lr:
        args.lr = adjust_learning_rate(optimizer, args.lr, args.weight_decay, epoch, args.epochs)

    if args.net == "fcn" or args.net == "psp":
        checkpoint_fn = os.path.join(args.pth_dir, "%s-%s-res%s-%s.pth.tar" % (
            args.savename, args.net, args.res, epoch + 1))
    else:
        checkpoint_fn = os.path.join(args.pth_dir, "%s-%s-%s.pth.tar" % (
            args.savename, args.net, epoch + 1))

    args.start_epoch = epoch + 1
    save_dic = {
        'args': args,
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }

    save_checkpoint(save_dic, is_best=False, filename=checkpoint_fn)
