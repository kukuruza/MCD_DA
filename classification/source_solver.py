from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from model.build_gen import *


# Training settings
class Solver(object):
    def __init__(self, args, batch_size=64, source='svhn',
                 target='mnist', learning_rate=0.0002, interval=100, optimizer='adam'
                 , num_k=4, all_use=False, checkpoint_dir=None, save_epoch=10):
        self.batch_size = batch_size
        self.source = source
        self.target = target
        self.num_k = num_k
        self.checkpoint_dir = checkpoint_dir
        self.save_epoch = save_epoch
        self.use_abs_diff = args.use_abs_diff
        self.all_use = all_use
        if self.source == 'svhn':
            self.scale = True
        else:
            self.scale = False
        print('dataset loading')
        if self.source == 'citycam' or self.target == 'citycam':
            import sys, os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'segmentation'))
            from transform import ReLabel, ToLabel, Scale, RandomSizedCrop, RandomHorizontalFlip, RandomRotation
            from PIL import Image
            from torchvision.transforms import Compose, Normalize, ToTensor
            from datasets import ConcatDataset, get_dataset, check_src_tgt_ok
            from models.model_util import get_models, get_optimizer

            train_img_shape = (64, 64)  #  tuple([int(x) for x in args.train_img_shape])
            img_transform_list = [
                Scale(train_img_shape, Image.BILINEAR),
                ToTensor(),
                Normalize([.485, .456, .406], [.229, .224, .225])
            ]
#            if args.augment:
#                aug_list = [
#                    RandomRotation(),
#                    RandomHorizontalFlip(),
#                    RandomSizedCrop()
#                ]
#                img_transform_list = aug_list + img_transform_list

            img_transform = Compose(img_transform_list)

            label_transform = Compose([
                Scale(train_img_shape, Image.NEAREST),
                ToLabel(),
                ReLabel(255, 12) # args.n_class - 1),  # Last Class is "Void" or "Background" class
            ])

            src_dataset_test = get_dataset(dataset_name='citycam',
                    split='synthetic-Sept19',
                    img_transform=img_transform, label_transform=label_transform,
                    test=True, input_ch=3, keys_dict={'image': 'image', 'yaw': 'label', 'yaw_raw': 'label_raw'})

            tgt_dataset_test = get_dataset(dataset_name='citycam',
                    split='real-Sept23-train, objectid IN (SELECT objectid FROM properties WHERE key="yaw")',
                    img_transform=img_transform, label_transform=label_transform,
                    test=True, input_ch=3, keys_dict={'image': 'image', 'yaw': 'label', 'yaw_raw': 'label_raw'})

            self.dataset_test = torch.utils.data.DataLoader(
                #src_dataset_test,
                tgt_dataset_test,
                batch_size=args.batch_size, shuffle=False,
                pin_memory=True)

            dataset_train = get_dataset(dataset_name='citycam',
                    split='synthetic-Sept19',
                    img_transform=img_transform, label_transform=label_transform,
                    test=False, input_ch=3, keys_dict={'image': 'S_image', 'yaw': 'S_label', 'yaw_raw': 'S_label_raw'})

            self.dataset_train = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=args.batch_size, shuffle=True,
                pin_memory=True)

        else: 
            from datasets_dir.dataset_read import dataset_read
            self.datasets_test, self.dataset_train = dataset_read(target, source, self.batch_size, scale=self.scale,
                                                            all_use=self.all_use)
        self.G = Generator(source=source, target=target)
        print('load finished!')
        self.C1 = Classifier(source=source, target=target)
        self.C2 = Classifier(source=source, target=target)
        if args.eval_only:
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (
                    self.checkpoint_dir, self.source, self.target, self.checkpoint_dir, args.resume_epoch))
            self.G.torch.load(
                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, args.resume_epoch))

        self.G.cuda()
        self.C1.cuda()
        self.C2.cuda()
        self.interval = interval

        self.set_optimizer(which_opt=optimizer, lr=learning_rate)
        self.lr = learning_rate

    def set_optimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)

            self.opt_c1 = optim.SGD(self.C1.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
            self.opt_c2 = optim.SGD(self.C2.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)

        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=lr, weight_decay=0.0005)

            self.opt_c1 = optim.Adam(self.C1.parameters(),
                                     lr=lr, weight_decay=0.0005)
            self.opt_c2 = optim.Adam(self.C2.parameters(),
                                     lr=lr, weight_decay=0.0005)

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()

    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))

    def discrepancy(self, out1, out2):
        return torch.mean(torch.abs(F.softmax(out1) - F.softmax(out2)))

    def train_onestep(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C1.train()
        self.C2.train()
        torch.cuda.manual_seed(1)

        for batch_idx, data in enumerate(self.dataset_train):
            img_s = data['S_image']
            label_s = data['S_label']
            if img_s.size()[0] < self.batch_size:
                break
            img_s = img_s.cuda()
            label_s = Variable(label_s.long().cuda())
            img_s = Variable(img_s)
            self.reset_grad()
            feat_s = self.G(img_s)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            loss_s.backward()#retain_variables=True)
            self.C1.set_lambda(1.0)
            self.C2.set_lambda(1.0)
            self.opt_c1.step()
            self.opt_c2.step()
            self.opt_g.step()
            self.reset_grad()
            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}'.format(
                    epoch, batch_idx, 100,
                    100. * batch_idx / 70000, loss_s1.data, loss_s2.data))
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s %s\n' % (loss_s1.data, loss_s2.data))
                    record.close()
        return batch_idx

    def test(self, epoch, record_file=None, save_model=False):
        self.G.eval()
        self.C1.eval()
        self.C2.eval()
        test_loss = 0
        correct1 = 0
        correct2 = 0
        correct3 = 0
        size = 0
        for batch_idx, data in enumerate(self.dataset_test):
            img = data['image']
            label = data['label']
            img, label = img.cuda(), label.long().cuda()
            with torch.no_grad():
                img, label = Variable(img), Variable(label)
            feat = self.G(img)
            output1 = self.C1(feat)
            output2 = self.C2(feat)
            test_loss += F.nll_loss(output1, label).data
            output_ensemble = output1 + output2
            pred1 = output1.data.max(1)[1]
            pred2 = output2.data.max(1)[1]
            pred_ensemble = output_ensemble.data.max(1)[1]
            k = label.data.size()[0]
            label = label.data
            correct1 += pred1.eq(label).cpu().sum()
            correct2 += pred2.eq(label).cpu().sum()
            correct3 += pred_ensemble.eq(label).cpu().sum()
            if self.source == 'citycam':
                nextup = (data['label_raw'] / 360. * 12. - label.cpu().double() > 0.5).long().cuda()
                label_next = torch.remainder(label - 1, 12) * (1. - nextup) \
                           + torch.remainder(label + 1, 12) * nextup
                correct1 += pred1.eq(label_next).cpu().sum()
                correct2 += pred2.eq(label_next).cpu().sum()
                correct3 += pred_ensemble.eq(label_next).cpu().sum()
            size += k
        test_loss = test_loss / size
        print(
            'Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%) Accuracy Ensemble: {}/{} ({:.0f}%) \n'.format(
                test_loss, correct1, size,
                100. * correct1 / size, correct2, size, 100. * correct2 / size, correct3, size, 100. * correct3 / size))
        if save_model and epoch % self.save_epoch == 0:
            torch.save(self.G,
                       '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
            torch.save(self.C1,
                       '%s/%s_to_%s_model_epoch%s_C1.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
            torch.save(self.C2,
                       '%s/%s_to_%s_model_epoch%s_C2.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
        if record_file:
            record = open(record_file, 'a')
            print('recording %s', record_file)
            record.write('%s %s %s\n' % (float(correct1) / size, float(correct2) / size, float(correct3) / size))
            record.close()
