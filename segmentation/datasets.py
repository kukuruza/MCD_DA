import collections
import glob
import os
import os.path as osp
import logging
from math import floor

import numpy as np
import torch
from PIL import Image
from PIL import ImageOps
from torch.utils import data

from transform import HorizontalFlip, VerticalFlip


class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, datasets):
        self.datasets = datasets
        assert len(self.datasets) == 2, 'For now 2 is enough'

        self.indlist2 = np.arange(len(self.datasets[1]))
        np.random.shuffle(self.indlist2)
        self.index2 = -1

    def increment_index2(self):
        if self.index2 == len(self.indlist2) - 1:
            np.random.shuffle(self.indlist2)
            self.index2 = 0
        else:
            self.index2 += 1

    def __getitem__(self, i):
        self.increment_index2()
        item = {}
        # Get item from source domain and add it to the new dictionary with  anew key.
        item_S = self.datasets[0][i]
        for key in item_S:
            item[key] = item_S[key]
        # Get item from target domain and add it to the new dictionary with  anew key.
        item_T = self.datasets[1][self.indlist2[self.index2]]
        for key in item_T:
            assert key not in item_S, (key, item_S.keys())
            item[key] = item_T[key]
        logging.debug('ConcatDataset: item.keys(): %s' % str(item.keys()))
        return item

    def __len__(self):
        return min(len(d) for d in self.datasets)


def default_loader(path):
    return Image.open(path)


class CityDataSet(data.Dataset):
    def __init__(self, root, split="train", img_transform=None, label_transform=None, test=True,
                 label_type=None, input_ch=3):
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.h_flip = HorizontalFlip()
        self.v_flip = VerticalFlip()
        self.test = test
        data_dir = root
        # for split in ["train", "trainval", "val"]:
        imgsets_dir = osp.join(data_dir, "leftImg8bit/%s.txt" % split)
        with open(imgsets_dir) as imgset_file:
            for name in imgset_file:
                name = name.strip()
                img_file = osp.join(data_dir, "leftImg8bit/%s" % name)
                if label_type == "label16":
                    name = name.replace('leftImg8bit', 'gtFine_label16IDs')
                else:
                    name = name.replace('leftImg8bit', 'gtFine_labelTrainIds')
                label_file = osp.join(data_dir, "gtFine/%s" % name)
                self.files[split].append({
                    "img": img_file,
                    "label": label_file
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')
        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return {'image': img, 'label_map': label, 'url': img_file}

        return {'image': img, 'label_map': label}


class GTADataSet(data.Dataset):
    def __init__(self, root, split="images", img_transform=None, label_transform=None,
                 test=False, input_ch=3):
        # Note; split "train" and "images" are SAME!!!

        assert split in ["images", "test", "train"]

        assert input_ch in [1, 3, 4]
        self.input_ch = input_ch
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.h_flip = HorizontalFlip()
        self.v_flip = VerticalFlip()
        self.test = test
        data_dir = root

        imgsets_dir = osp.join(data_dir, "%s.txt" % split)
        with open(imgsets_dir) as imgset_file:
            for name in imgset_file:
                name = name.strip()
                img_file = osp.join(data_dir, "%s" % name)
                label_file = osp.join(data_dir, "%s" % name.replace('images', 'labels_gt'))
                self.files[split].append({
                    "img": img_file,
                    "label": label_file
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')
        np3ch = np.array(img)
        if self.input_ch == 1:
            img = ImageOps.grayscale(img)

        elif self.input_ch == 4:
            extended_np3ch = np.concatenate([np3ch, np3ch[:, :, 0:1]], axis=2)
            img = Image.fromarray(np.uint8(extended_np3ch))

        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return {'image': img, 'label_map': label, 'url': img_file}

        return {'image': img, 'label_map': label}


class SynthiaDataSet(data.Dataset):
    def __init__(self, root, split="all", img_transform=None, label_transform=None,
                 test=False, input_ch=3):
        # TODO this does not support "split" parameter

        assert input_ch in [1, 3, 4]
        self.input_ch = input_ch
        self.root = root
        self.split = split
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.test = test

        rgb_dir = osp.join(root, "RGB")
        gt_dir = osp.join(root, "GT", "LABELS16")

        rgb_fn_list = glob.glob(osp.join(rgb_dir, "*.png"))
        gt_fn_list = glob.glob(osp.join(gt_dir, "*.png"))

        for rgb_fn, gt_fn in zip(rgb_fn_list, gt_fn_list):
            self.files[split].append({
                "rgb": rgb_fn,
                "label": gt_fn
            })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]
        img_file = datafiles["rgb"]
        img = Image.open(img_file).convert('RGB')
        label_file = datafiles["label"]
        label = Image.open(label_file).convert("P")

        if self.img_transform:
            img = self.img_transform(img)

        if self.label_transform:
            label = self.label_transform(label)

        if self.test:
            return {'image': img, 'label_map': label, 'url': img_file}

        return {'image': img, 'label_map': label}


class CitycamDataSet(data.Dataset):

    def _onehot_yaw(self, yaw):
        assert yaw >= 0 and yaw < 360., yaw
        yaw = yaw / 360. * 12.

        onehot = np.zeros((12,), dtype=float)
        integral = int(floor(yaw))
        fraction = yaw - floor(yaw)
        if fraction < 0.5:
            onehot[integral] = 1. - fraction
            onehot[(integral - 1) % 12] = fraction
        else:
            onehot[integral] = fraction
            onehot[(integral + 1) % 12] = 1. - fraction
        return onehot

    def __init__(self, root, split, img_transform=None, label_transform=None,
                test=False, input_ch=3, 
                keys_dict={}):
        ''' 
        Args:
            split:      The name of the db in segmentation/data/citycam without ext.
                        After a comma, the WHERE clause for objects can be specified.
            keys_dict:  Necessary keys and their names for gititem.
        '''
        import os, sys
        sys.path.insert(0, os.path.join(os.getenv('HOME'), 'projects/shuffler/lib'))
        from interfacePytorch import ImagesDataset, ObjectsDataset
        from backendDb import objectField

        # Parse info from split.
        split_list = split.split(',')
        split = split_list[0]
        where_object = split_list[1] if len(split_list) > 1 else 'TRUE'

        self.img_transform = img_transform
        self.label_transform = label_transform

        logging.info('Dataset root: %s, split: %s' % (split, root))
        db_file = os.path.realpath(os.path.join(root, split + '.db'))
        logging.info('db_file is resolved to "%s"' % db_file)
        assert os.path.exists(db_file)
        self.dataset = ObjectsDataset(db_file=db_file, rootdir=os.getenv('CITY_PATH'), where_object=where_object)
        self.size = len(self.dataset)

        self.keys_dict = keys_dict

    def __getitem__(self, index):
        car = self.dataset[index]

        image_original = car['image'].copy()  # Save the original.
        if self.img_transform:
            image = Image.fromarray(image_original)
            image = self.img_transform(image)

        #print (item)
        item = {}
        if 'image' in self.keys_dict:
          item[self.keys_dict['image']] = image
        if 'url' in self.keys_dict:
          item[self.keys_dict['url']] = car['imagefile']
        if 'image_original' in self.keys_dict:
          item[self.keys_dict['image_original']] = image_original

        if 'mask' in self.keys_dict:
            mask = car['mask']
            assert mask is not None
            mask = (mask < 128).astype(np.uint8) * 255  # Background is 255 now.
            if self.label_transform:
                mask = Image.fromarray(mask).convert("P")
                mask = self.label_transform(mask)
            item[self.keys_dict['mask']] = mask

        if 'yaw' in self.keys_dict:
            yaw = float(car['yaw'])
            yaw_discr = int(floor(yaw / 360 * 12 + 0.5)) % 12
            logging.debug('Yaw %1.f transformed into one-hot %s' % (yaw, str(yaw_discr)))
            if 'yaw' in self.keys_dict:
              item[self.keys_dict['yaw']] = yaw_discr
            if 'yaw_onehot' in self.keys_dict:
              item[self.keys_dict['yaw_onehot']] = self._onehot_yaw(yaw)
            if 'yaw_raw' in self.keys_dict:
              item[self.keys_dict['yaw_raw']] = yaw % 360

        if 'pitch' in self.keys_dict:
            pitch = float(car['pitch'])
            pitch /= 90.
            item[self.keys_dict['pitch']] = pitch

        return item

    def __len__(self):
        return self.size


class TestDataSet(data.Dataset):
    def __init__(self, root, split="train", img_transform=None, label_transform=None, test=True, input_ch=3):
        assert input_ch == 3
        self.root = root
        self.split = split
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.files = collections.defaultdict(list)
        self.img_transform = img_transform
        self.label_transform = label_transform
        self.h_flip = HorizontalFlip()
        self.v_flip = VerticalFlip()
        self.test = test
        data_dir = root
        # for split in ["train", "trainval", "val"]:
        imgsets_dir = os.listdir(data_dir)
        for name in imgsets_dir:
            img_file = osp.join(data_dir, "%s" % name)
            self.files[split].append({
                "img": img_file,
            })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        datafiles = self.files[self.split][index]

        img_file = datafiles["img"]
        img = Image.open(img_file).convert('RGB')

        if self.img_transform:
            img = self.img_transform(img)

        if self.test:
            return {'image': img, 'label_map': 'hoge', 'url': img_file}

        return {'image': img, 'label_map': img}


def get_dataset(dataset_name, split, img_transform, label_transform, test, input_ch, **kwargs):
    assert dataset_name in ["gta", "city", "test", "city16", "synthia", "citycam"]

    name2obj = {
        "gta": GTADataSet,
        "city": CityDataSet,
        "city16": CityDataSet,
        "synthia": SynthiaDataSet,
        "citycam": CitycamDataSet,
    }
    ##Note fill in the blank below !! "gta....fill the directory over images folder.
    name2root = {
        "gta": "data/gta5",  ## Fill the directory over images folder. put train.txt, val.txt in this folder
        "city": "data/cityscrapes",  ## ex, ./www.cityscapes-dataset.com/file-handling
        "city16": "",  ## Same as city
        "synthia": "",  ## synthia/RAND_CITYSCAPES",
        "citycam": "data/citycam",
    }
    dataset_obj = name2obj[dataset_name]
    root = name2root[dataset_name]

    if dataset_name == "city16":
        dataset = dataset_obj(root=root, split=split, img_transform=img_transform, label_transform=label_transform,
                           test=test, input_ch=input_ch, label_type="label16")
    else:
        dataset = dataset_obj(root=root, split=split, img_transform=img_transform, label_transform=label_transform,
                           test=test, input_ch=input_ch, **kwargs)
    print ('Initialized dataset "%s" with split "%s" and size %d' % (dataset_name, split, len(dataset)))
    return dataset


def check_src_tgt_ok(src_dataset_name, tgt_dataset_name):
    if src_dataset_name == "synthia" and not tgt_dataset_name == "city16":
        raise AssertionError("you must use synthia-city16 pair")
    elif src_dataset_name == "city16" and not tgt_dataset_name == "synthia":
        raise AssertionError("you must use synthia-city16 pair")
    elif "citycam" in src_dataset_name != "citycam" in tgt_dataset_name:
        raise AssertionError("Citycam are either both or none of src and tgt")


def get_n_class(src_dataset_name):
    if src_dataset_name in ["synthia", "city16"]:
        return 16
    elif src_dataset_name in ["gta", "city", "test"]:
        return 20
    elif src_dataset_name in ["citycam"]:
        return 2
    else:
        raise NotImplementedError("You have to define the class of %s dataset" % src_dataset_name)
