# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
from PIL import Image, ImageEnhance
import random

import box_utils


def random_distort(img):
    def random_brightness(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Brightness(img).enhance(e)
 
    def random_contrast(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Contrast(img).enhance(e)
 
    def random_color(img, lower=0.5, upper=1.5):
        e = np.random.uniform(lower, upper)
        return ImageEnhance.Color(img).enhance(e)
 
    ops = [random_brightness, random_contrast, random_color]
    np.random.shuffle(ops)
 
    img = Image.fromarray(img)
    img = ops[0](img)
    img = ops[1](img)
    img = ops[2](img)
    img = np.asarray(img)
 
    return img


def random_crop(img, boxes, labels, scales=[0.3, 1.0], max_ratio=2.0, constraints=None, max_trial=50):
    if len(boxes) == 0:
        return img, boxes

    if not constraints:
        constraints = [
                (0.1, 1.0),
                (0.3, 1.0),
                (0.5, 1.0),
                (0.7, 1.0),
                (0.9, 1.0),
                (0.0, 1.0)]

    img = Image.fromarray(img)
    w, h = map(float, img.size)
    crops = [(0, 0, w, h)]
    for min_iou, max_iou in constraints:
        for _ in range(max_trial):
            scale = random.uniform(scales[0], scales[1])
            aspect_ratio = random.uniform(max(1 / max_ratio, scale * scale), \
                                          min(max_ratio, 1 / scale / scale))
            crop_h = int(h * scale / np.sqrt(aspect_ratio))
            crop_w = int(w * scale * np.sqrt(aspect_ratio))
            crop_x = random.randrange(w - crop_w)
            crop_y = random.randrange(h - crop_h)
            crop_box = np.array([[
                (crop_x + crop_w / 2.0) / w,
                (crop_y + crop_h / 2.0) / h,
                crop_w / w,
                crop_h /h
                ]])

            iou = box_utils.box_iou_xywh(crop_box, boxes)
            if min_iou <= iou.min() and max_iou >= iou.max():
                crops.append((crop_x, crop_y, crop_w, crop_h))
                break

    while crops:
        crop = crops.pop(np.random.randint(0, len(crops)))
        crop_boxes, crop_labels, box_num = box_utils.box_crop(boxes, labels, crop, (w, h))
        if box_num < 1:
            continue
        img = img.crop((crop[0], crop[1], crop[0] + crop[2], crop[1] + crop[3])).resize(img.size, Image.LANCZOS)
        img = np.asarray(img)
        return img, crop_boxes, crop_labels
    img = np.asarray(img)
    return img, boxes, labels

def random_flip(img, gtboxes, thresh=0.5):
    if random.random() > thresh:
        img = img[:, ::-1, :]
        gtboxes[:, 0] = 1.0 - gtboxes[:, 0]
    return img, gtboxes

def random_interp(img, size):
    interp_method = [
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_AREA,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
        ]
    interp = interp_method[random.randint(0, len(interp_method) - 1)]
    h, w, _ = img.shape
    im_scale_x = size / float(w)
    im_scale_y = size / float(h)
    img = cv2.resize(img, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=interp)
    return img

def random_expand(img, gtboxes, max_ratio=4., fill=None, keep_ratio=True, thresh=0.5):
    if random.random() > thresh:
        return img, gtboxes

    if max_ratio < 1.0:
        return img, gtboxes
    
    h, w, c = img.shape
    ratio_x = random.uniform(1, max_ratio)
    if keep_ratio:
        ratio_y = ratio_x
    else:
        ratio_y = random.uniform(1, max_ratio)
    oh = int(h * ratio_y)
    ow = int(w * ratio_x)
    off_x = random.randint(0, ow -w)
    off_y = random.randint(0, oh -h)

    out_img = np.zeros((oh, ow, c))
    if fill and len(fill) == c:
        for i in range(c):
            out_img[:, :, i] = fill[i] * 255.0

    out_img[off_y: off_y + h, off_x: off_x + w, :] = img
    gtboxes[:, 0] = ((gtboxes[:, 0] * w) + off_x) / float(ow)
    gtboxes[:, 1] = ((gtboxes[:, 1] * h) + off_y) / float(oh)
    gtboxes[:, 2] = gtboxes[:, 2] / ratio_x
    gtboxes[:, 3] = gtboxes[:, 3] / ratio_y

    return out_img.astype('uint8'), gtboxes

def image_mixup(img1, gtboxes1, gtlabels1, img2, gtboxes2, gtlabels2):
    factor = np.random.beta(1.5, 1.5)
    if factor >= 1.0:
        return img1, gtboxes1, gtlabels1

    h = max(img1.shape[0], img2.shape[0])
    w = max(img1.shape[1], img2.shape[1])
    img = np.zeros((h, w, img1.shape[2]), 'float32')
    img[:img1.shape[0], :img1.shape[1], :] = img1.astype('float32') * factor
    img[:img2.shape[0], :img2.shape[1], :] += img2.astype('float32') * (1.0 - factor)
    gtboxes = np.zeros_like(gtboxes1)
    gtlabels = np.zeros_like(gtlabels1)

    gt_valid_mask1 = np.logical_and(gtboxes1[:, 2] > 0, gtboxes1[:, 3] > 0)
    gtboxes1 = gtboxes1[gt_valid_mask1]
    gtlabels1 = gtlabels1[gt_valid_mask1]
    gtboxes1[:, 0] = gtboxes1[:, 0] * img1.shape[1] / w
    gtboxes1[:, 1] = gtboxes1[:, 1] * img1.shape[0] / h
    gtboxes1[:, 2] = gtboxes1[:, 2] * img1.shape[1] / w
    gtboxes1[:, 3] = gtboxes1[:, 3] * img1.shape[0] / h

    gt_valid_mask2 = np.logical_and(gtboxes2[:, 2] > 0, gtboxes2[:, 3] > 0)
    gtboxes2 = gtboxes2[gt_valid_mask2]
    gtlabels2 = gtlabels2[gt_valid_mask2]
    gtboxes2[:, 0] = gtboxes2[:, 0] * img2.shape[1] / w
    gtboxes2[:, 1] = gtboxes2[:, 1] * img2.shape[0] / h
    gtboxes2[:, 2] = gtboxes2[:, 2] * img2.shape[1] / w
    gtboxes2[:, 3] = gtboxes2[:, 3] * img2.shape[0] / h
    gtboxes_all = np.concatenate((gtboxes1, gtboxes2), axis=0)
    gtlabels_all = np.concatenate((gtlabels1, gtlabels2), axis=0)
    gt_num = min(len(gtboxes), len(gtboxes_all))
    gtboxes[:gt_num] = gtboxes_all[:gt_num]
    gtlabels[:gt_num] = gtlabels_all[:gt_num]
    return img.astype('uint8'), gtboxes, gtlabels

def image_augment(img, gtboxes, gtlabels, size, means=None):
    img = random_distort(img)
    img, gtboxes = random_expand(img, gtboxes, fill=means)
    img, gtboxes, gtlabels = random_crop(img, gtboxes, gtlabels)
    img = random_interp(img, size)
    img, gtboxes = random_flip(img, gtboxes)

    return img.astype('float32'), gtboxes.astype('float32'), gtlabels.astype('int32')

