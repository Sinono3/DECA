# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os, sys
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
import scipy.io

from . import detectors


def video2sequence(video_path, sample_step=10):
    videofolder = os.path.splitext(video_path)[0]
    os.makedirs(videofolder, exist_ok=True)
    video_name = os.path.splitext(os.path.split(video_path)[-1])[0]
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        # if count%sample_step == 0:
        imagepath = os.path.join(videofolder, f"{video_name}_frame{count:04d}.jpg")
        cv2.imwrite(imagepath, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print("video frames are stored in {}".format(videofolder))
    return imagepath_list


def bbox2point(left, right, top, bottom, type="bbox"):
    """bbox from detector and landmarks are different"""
    if type == "kpt68":
        old_size = (right - left + bottom - top) / 2 * 1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
    elif type == "bbox":
        old_size = (right - left + bottom - top) / 2
        center = np.array(
            [
                right - (right - left) / 2.0,
                bottom - (bottom - top) / 2.0 + old_size * 0.12,
            ]
        )
    else:
        raise NotImplementedError
    return old_size, center


class TestData(Dataset):
    def __init__(
        self,
        testpath,
        face_detector,
        iscrop=True,
        crop_size=224,
        scale=1.25,
        sample_step=10,
    ):
        """
        testpath: folder, imagepath_list, image path, video path
        """
        self.imagepath_list = testpath

        # print('total {} images'.format(len(self.imagepath_list)))
        self.crop_size = crop_size
        self.scale = scale
        self.iscrop = iscrop
        self.resolution_inp = crop_size
        if face_detector is not None:
            self.face_detector = face_detector
        else:
            print(f"please check the detector: {face_detector}")
            exit()

    def __len__(self):
        return len(self.imagepath_list)

    def __getitem__(self, index):
        imagepath = self.imagepath_list[index]
        imagename = os.path.splitext(os.path.split(imagepath)[-1])[0]
        image = np.array(imread(imagepath))
        return TestData.img_to_td(
            image, self.face_detector, imagename=imagename, iscrop=self.iscrop
        )

    # Expects a numpy in cv2.imread format
    def img_to_td(
        image: np.ndarray,
        face_detector,
        imagename="",
        iscrop=True,
        crop_size=224,
        scale=1.25,
    ):
        h, w, _ = image.shape
        if iscrop:
            bbox, bbox_type = face_detector.run(image)
            if len(bbox) < 4:
                print("no face detected! run original image")
                left, right, top, bottom = (0, h - 1, 0, w - 1)
            else:
                left, right, top, bottom = (bbox[0], bbox[2], bbox[1], bbox[3])
            old_size, center = bbox2point(left, right, top, bottom, type=bbox_type)
            size = int(old_size * scale)
            src_pts = np.array(
                [
                    [center[0] - size / 2, center[1] - size / 2],
                    [center[0] - size / 2, center[1] + size / 2],
                    [center[0] + size / 2, center[1] - size / 2],
                ]
            )
        else:
            src_pts = np.array([[0, 0], [0, h - 1], [w - 1, 0]])

        DST_PTS = np.array([[0, 0], [0, crop_size - 1], [crop_size - 1, 0]])
        tform = estimate_transform("similarity", src_pts, DST_PTS)

        image = image / 255.0

        dst_image = warp(image, tform.inverse, output_shape=(crop_size, crop_size))
        dst_image = dst_image.transpose(2, 0, 1)
        return {
            "image": torch.tensor(dst_image).float(),
            "imagename": imagename,
            "tform": torch.tensor(tform.params).float(),
            "original_image": torch.tensor(image.transpose(2, 0, 1)).float(),
        }
