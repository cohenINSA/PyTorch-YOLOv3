#!/usr/bin/python
# encoding: utf-8
import random
import os
from PIL import Image
import numpy as np


def scale_image_channel(im, c, v):
    cs = list(im.split())
    cs[c] = cs[c].point(lambda i: i * v)
    out = Image.merge(im.mode, tuple(cs))
    return out


def distort_image(im, hue, sat, val):
    im = im.convert('HSV')
    cs = list(im.split())
    cs[1] = cs[1].point(lambda i: i * sat)
    cs[2] = cs[2].point(lambda i: i * val)

    def change_hue(x):
        x += hue * 255
        if x > 255:
            x -= 255
        if x < 0:
            x += 255
        return x

    cs[0] = cs[0].point(change_hue)
    im = Image.merge(im.mode, tuple(cs))

    im = im.convert('RGB')
    # constrain_image(im)
    return im


def rand_scale(s):
    scale = random.uniform(1, s)
    if (random.randint(1, 10000) % 2):
        return scale
    return 1. / scale


def random_distort_image(im, hue, saturation, exposure):
    dhue = random.uniform(-hue, hue)
    dsat = rand_scale(saturation)
    dexp = rand_scale(exposure)
    res = distort_image(im, dhue, dsat, dexp)
    return res


def data_augmentation(img, shape, jitter, hue, saturation, exposure, flip):
    oh = img.height
    ow = img.width

    dw = int(ow * jitter)
    dh = int(oh * jitter)

    pleft = random.randint(-dw, dw)
    pright = random.randint(-dw, dw)
    ptop = random.randint(-dh, dh)
    pbot = random.randint(-dh, dh)

    swidth = ow - pleft - pright
    sheight = oh - ptop - pbot

    sx = float(swidth) / ow
    sy = float(sheight) / oh

    cropped = img.crop((pleft, ptop, pleft + swidth - 1, ptop + sheight - 1))

    dx = (float(pleft) / ow) / sx
    dy = (float(ptop) / oh) / sy

    if shape is None:
        shape = (ow, oh)
    sized = cropped.resize(shape)

    if flip:
        sized = sized.transpose(Image.FLIP_LEFT_RIGHT)
    img = random_distort_image(sized, hue, saturation, exposure)

    return img, flip, dx, dy, sx, sy


def fill_truth_detection(labpath, w, h, flip, dx, dy, sx, sy):
    label = np.zeros((0, 5))
    if os.path.getsize(labpath):
        try:
            bs = np.loadtxt(labpath).reshape(-1, 5)
        except UserWarning as e:
            print("User Warning: {}".format(e))
            return label
        except Exception:
            labels = np.zeros((0, 5))
        label = np.zeros((len(bs), 5))
        cc = 0
        for i in range(bs.shape[0]):  # process the rows
            x1 = bs[i][1] - bs[i][3] / 2
            y1 = bs[i][2] - bs[i][4] / 2
            x2 = bs[i][1] + bs[i][3] / 2
            y2 = bs[i][2] + bs[i][4] / 2

            x1 = min(0.999, max(0, x1 * sx - dx))
            y1 = min(0.999, max(0, y1 * sy - dy))
            x2 = min(0.999, max(0, x2 * sx - dx))
            y2 = min(0.999, max(0, y2 * sy - dy))

            bs[i][1] = (x1 + x2) / 2
            bs[i][2] = (y1 + y2) / 2
            bs[i][3] = (x2 - x1)
            bs[i][4] = (y2 - y1)

            if flip:
                bs[i][1] = 0.999 - bs[i][1]

            if bs[i][3] < 0.001 or bs[i][4] < 0.001:
                continue
            label[cc] = bs[i]

            cc += 1
            if cc >= 50:
                break

    # label = np.reshape(label, (-1))
    return label


def load_data_detection(imgpath, shape, jitter, hue, saturation, exposure, flip):
    '''
    Return the CENTER coordinates of the bounding box, the width and the height, taking into account the image
    augmentation parameters
    '''
    labpath = imgpath.replace('.jpg', '.txt').replace('.png','.txt')
    #labpath = imgpath.replace('images', 'labels').replace('JPEGImages', 'labels').replace('.jpg', '.txt').replace('.png','.txt')

    ## data augmentation
    img = Image.open(imgpath).convert('RGB')
    img, flip, dx, dy, sx, sy = data_augmentation(img, shape, jitter, hue, saturation, exposure, flip)
    label = fill_truth_detection(labpath, img.width, img.height, flip, dx, dy, 1./sx, 1./sy)
    return img, label


def shift_to_xy(label):
    """
    Changes label from center coordinates to top-left and bottom-right coordinates.
    First-element of label is the class.
    """
    new_label = np.zeros(label.shape)
    for i in range(label.shape[0]):  # process the rows
        new_label[i][0] = label[i][0]
        new_label[i][1] = label[i][1] - label[i][3] / 2
        new_label[i][2] = label[i][2] - label[i][4] / 2
        new_label[i][3] = label[i][1] + label[i][3] / 2
        new_label[i][4] = label[i][2] + label[i][4] / 2
    return new_label


def rescale_full_range(label, w, h):
    """
    Label first element is the class. Other elements are [x1, y1, x2, y2]
    """
    for i in range(label.shape[0]):
        # print(label[i])
        # print(label[i][1]*w)
        label[i][1] = int(label[i][1]*w)
        label[i][2] = int(label[i][2]*h)
        label[i][3] = int(label[i][3]*w)
        label[i][4] = int(label[i][4]*h)
    return label
