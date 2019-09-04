import torch
import torch.nn.functional as F
import numpy as np


def horizontal_flip(images, targets):
    images = torch.flip(images, [-1])
    targets[:, 2] = 1 - targets[:, 2]
    return images, targets


def horizontal_flip_no_cls(images, targets_no_cls):
    images = torch.flip(images, [-1])
    targets_no_cls[:, 0] = 1 - targets_no_cls[:, 0]  # change only x coordinate
    return images, targets_no_cls
