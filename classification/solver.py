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

            train_img_shape = (32, 32)  #  tuple([int(x) for x in args.train_img_shape])
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

            src_dataset = get_dataset(dataset_name='citycam',
                    split='synthetic-w132-goodtypes',
                    img_transform=img_transform, label_transform=label_transform,
                    test=False, input_ch=3)

            tgt_dataset = get_dataset(dataset_name='citycam',
                    split='real-w64',
                    img_transform=img_transform, label_transform=label_transform,
                    test=False, input_ch=3)

            self.datasets = torch.utils.data.DataLoader(
                ConcatDataset([src_dataset, tgt_dataset],
                    S_keys=['image', 'label'], T_keys=['image']
                    ),
                batch_size=args.batch_size, shuffle=True,
                pin_memory=True)

            dataset_test = get_dataset(dataset_name='citycam',
#                    split='synthetic-w132-goodtypes',
                    split='real-w64, 1, yaw IS NOT NULL',
                    img_transform=img_transform, label_transform=label_transform,
                    test=False, input_ch=3)

            self.dataset_test = torch.utils.data.DataLoader(
                dataset_test,
                batch_size=args.batch_size, shuffle=True,
                pin_memory=True)

        #    self.G, _, _ = get_models(net_name='drn_d_105', input_ch=3,
        #                                     n_class=12,
        #                                     is_data_parallel=False)
        else: 
            from datasets_dir.dataset_read import dataset_read
            self.datasets, self.dataset_test = dataset_read(source, target, self.batch_size, scale=self.scale,
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

    def train(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C1.train()
        self.C2.train()
        torch.cuda.manual_seed(1)

        for batch_idx, data in enumerate(self.datasets):
            img_t = data['T_image']
            img_s = data['S_image']
            label_s = data['S_label']
            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break
            img_s = img_s.cuda()
            img_t = img_t.cuda()
            imgs = Variable(torch.cat((img_s, \
                                       img_t), 0))
            label_s = Variable(label_s.long().cuda())

            img_s = Variable(img_s)
            img_t = Variable(img_t)
            self.reset_grad()
            feat_s = self.G(img_s)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)

            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            loss_s.backward()
            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            feat_s = self.G(img_s)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            feat_t = self.G(img_t)
            output_t1 = self.C1(feat_t)
            output_t2 = self.C2(feat_t)

            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            loss_dis = self.discrepancy(output_t1, output_t2)
            loss = loss_s - loss_dis
            loss.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            for i in xrange(self.num_k):
                #
                feat_t = self.G(img_t)
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                loss_dis = self.discrepancy(output_t1, output_t2)
                loss_dis.backward()
                self.opt_g.step()
                self.reset_grad()
            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
                    epoch, batch_idx, 100,
                    100. * batch_idx / 70000, loss_s1.item(), loss_s2.item(), loss_dis.item()))
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s %s %s\n' % (loss_dis.item(), loss_s1.item(), loss_s2.item()))
                    record.close()
        return batch_idx

    def train_onestep(self, epoch, record_file=None):
        criterion = nn.CrossEntropyLoss().cuda()
        self.G.train()
        self.C1.train()
        self.C2.train()
        torch.cuda.manual_seed(1)

        for batch_idx, data in enumerate(self.datasets):
            img_t = data['T_image']
            img_s = data['S_image']
            label_s = data['S_label']
            if img_s.size()[0] < self.batch_size or img_t.size()[0] < self.batch_size:
                break
            img_s = img_s.cuda()
            img_t = img_t.cuda()
            label_s = Variable(label_s.long().cuda())
            img_s = Variable(img_s)
            img_t = Variable(img_t)
            self.reset_grad()
            feat_s = self.G(img_s)
            #print ('feat_s.size()', feat_s.size())
            #feat_s = feat_s.view(-1, 3072)
#            print ('feat_s.size()', feat_s.size())
            #assert 0
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
#            print ('output_s1.size()', output_s1.size())
#            print ('label_s.size()', label_s.size())
#            print (label_s)
            loss_s1 = criterion(output_s1, label_s)
            loss_s2 = criterion(output_s2, label_s)
            loss_s = loss_s1 + loss_s2
            loss_s.backward()#retain_variables=True)
            feat_t = self.G(img_t)
            self.C1.set_lambda(1.0)
            self.C2.set_lambda(1.0)
            output_t1 = self.C1(feat_t, reverse=True)
            output_t2 = self.C2(feat_t, reverse=True)
            loss_dis = -self.discrepancy(output_t1, output_t2)
            #loss_dis.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            self.opt_g.step()
            self.reset_grad()
            if batch_idx > 500:
                return batch_idx

            if batch_idx % self.interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
                    epoch, batch_idx, 100,
                    100. * batch_idx / 70000, loss_s1.data, loss_s2.data, loss_dis.data))
                if record_file:
                    record = open(record_file, 'a')
                    record.write('%s %s %s\n' % (loss_dis.data, loss_s1.data, loss_s2.data))
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
            test_loss += F.nll_loss(output1, label).data[0]
            output_ensemble = output1 + output2
            pred1 = output1.data.max(1)[1]
            pred2 = output2.data.max(1)[1]
            pred_ensemble = output_ensemble.data.max(1)[1]
            k = label.data.size()[0]
            correct1 += pred1.eq(label.data).cpu().sum()
            correct2 += pred2.eq(label.data).cpu().sum()
            correct3 += pred_ensemble.eq(label.data).cpu().sum()
            size += k
            for m in range(10):
                print ('%1.f' % float(data['label_raw'][m].numpy()), label.cpu().data[m].numpy(), pred1.data[m].cpu().numpy())
        test_loss = test_loss / size
        print(
            '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%) Accuracy Ensemble: {}/{} ({:.0f}%) \n'.format(
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
