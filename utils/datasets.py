import glob
import random
import os
import sys
import numpy as np
from PIL import Image
from PIL import ImageDraw
import torch
import torch.nn.functional as F

from utils.augmentations import *
from utils.image import *
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def random_resize(images, min_size=288, max_size=448):
    new_size = random.sample(list(range(min_size, max_size + 1, 32)), 1)[0]
    images = F.interpolate(images, size=new_size, mode="nearest")
    return images


class ImageFolder(Dataset):
    def __init__(self, folder_path, img_size=416):
        self.files = sorted(glob.glob("%s/*.*" % folder_path))
        self.img_size = img_size

    def __getitem__(self, index):
        img_path = self.files[index % len(self.files)]
        # Extract image as PyTorch tensor
        img = transforms.ToTensor()(Image.open(img_path))
        # Pad to square resolution
        img, _ = pad_to_square(img, 0)
        # Resize
        img = resize(img, self.img_size)

        return img_path, img

    def __len__(self):
        return len(self.files)


class ListDataset(Dataset):
    def __init__(self, list_path, img_size=416, multiscale=True, transform=None, train=True,
                 target_transform=None, data_augmentation=None, shuffle=True):
        with open(list_path, "r") as file:
            self.img_files = file.readlines()

        self.label_files = [
            path.replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.Nsamples = len(self.img_files)
        self.shape = img_size
        self.max_objects = 100
        self.transform = transform
        self.target_transform = target_transform
        self.data_augmentation = data_augmentation
        self.multiscale = multiscale
        self.min_size = self.shape - 3 * 32
        self.max_size = self.shape + 3 * 32
        self.batch_count = 0
        self.train = train

    def __getitem__(self, index):
        assert index < len(self), 'index range error'

        img_path = self.img_files[index % len(self.img_files)].rstrip()
        if self.train:
            jitter = self.data_augmentation['jitter']
            hue = self.data_augmentation['hue']
            saturation = self.data_augmentation['saturation']
            exposure = self.data_augmentation['exposure']
            flip = self.data_augmentation['flip']

            img, labels = load_data_detection(img_path, (self.shape, self.shape), jitter, hue, saturation, exposure, flip)

            if len(labels) == 0:
                labels = np.zeros((0, 5))

        else:
            img = Image.open(img_path).convert('RGB')
            if self.shape:
                img = img.resize((self.shape, self.shape))

            labpath = img_path.replace('.jpg', '.txt').replace('.png', '.txt')
            try:
                labels = np.loadtxt(labpath).reshape(-1, 5)  # if empty, return array([], shape=(0, 5), dtype=float64)
            except UserWarning as e:
                print("User Warning: {}".format(e))
            except Exception:
                labels = np.zeros((0, 5))

        labels = torch.from_numpy(labels)  # (n_boxes, 5)

        targets = torch.zeros((len(labels), 6))
        targets[:, 1:] = labels

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.shape = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.shape) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return self.Nsamples


class ListDatasetFasterRCNN(Dataset):
    def __init__(self, list_path, img_size=(416, 416), transform=None, train=True,
                 target_transform=None, data_augmentation=None, shuffle=True):
        with open(list_path, "r") as file:
            img_files = file.readlines()
        self.img_files = []

        # Remove empty images
        for img in img_files:
            with open(img.replace(".png", ".txt").replace(".jpg", ".txt").strip("\n"), "r") as file:
                lines = file.readlines()
            if len(lines) > 0:
                for l in lines:
                    if len(l) > 0:
                        self.img_files.append(img)

        self.label_files = [
            path.replace(".png", ".txt").replace(".jpg", ".txt")
            for path in self.img_files
        ]
        self.Nsamples = len(self.img_files)
        self.shape = img_size
        self.max_objects = 100
        self.batch_count = 0
        self.transform = transform
        self.target_transform = target_transform
        self.data_augmentation = data_augmentation
        self.train = train

    def __getitem__(self, index):
        assert index < len(self), 'index range error'

        outputs = dict()
        img_path = self.img_files[index % len(self.img_files)].rstrip()
        if self.train:
            jitter = self.data_augmentation['jitter']
            hue = self.data_augmentation['hue']
            saturation = self.data_augmentation['saturation']
            exposure = self.data_augmentation['exposure']
            flip = self.data_augmentation['flip']

            img, labels = load_data_detection(img_path, self.shape, jitter, hue, saturation, exposure, flip)

            labels = shift_to_xy(labels)
            if len(labels) > 0:
                labels_full_range = rescale_full_range(labels, img.width, img.height)
                labels = torch.from_numpy(labels)
                labels_full_range = torch.from_numpy(labels_full_range)
            else:
                labels_full_range = torch.from_numpy(labels)
        else:
            img = Image.open(img_path).convert('RGB')
            if self.shape:
                img = img.resize(self.shape)

            labpath = img_path.replace('.jpg', '.txt').replace('.png', '.txt')
            try:
                labels = np.loadtxt(labpath).reshape(-1, 5)  # if empty, return array([], shape=(0, 5), dtype=float64)
            except UserWarning as e:
                print("User Warning: {}".format(e))
            except Exception:
                labels = np.zeros((0, 5))

            labels = shift_to_xy(labels)
            labels_full_range = rescale_full_range(labels, img.width, img.height)
            labels = torch.from_numpy(labels)
            labels_full_range = torch.from_numpy(labels_full_range)

        boxes_only = torch.zeros((len(labels_full_range), 4), dtype=torch.float32)
        boxes_only[:] = labels_full_range[:, 1:]
        boxes_only = labels_full_range[:, 1:]

        cls_only = torch.zeros(len(labels_full_range), dtype=torch.int64)
        cls_only[:] = labels_full_range[:, 0] + 1

        outputs['boxes'] = boxes_only.float()
        outputs['labels'] = cls_only
        outputs['image_id'] = torch.tensor([index])
        outputs['iscrowd'] = torch.zeros((len(labels_full_range),), dtype=torch.int64)

        area = (boxes_only[:, 3]-boxes_only[:, 1]) * (boxes_only[:, 2] - boxes_only[:, 0])

        outputs['area'] = area

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            outputs['boxes'] = self.target_transform(outputs['boxes'])

        return img_path, img, outputs

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [target for target in targets]
        # # Add sample index to targets
        # for i, boxes in enumerate(targets):
        #     boxes[:, 0] = i
        # targets = torch.cat(targets, 0)

        # Resize images to input shape
        # imgs = torch.stack([resize(img, self.shape) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets

    def __len__(self):
        return self.Nsamples

    def get_height_and_width(self):
        return self.shape
