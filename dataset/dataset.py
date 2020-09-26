import os
import glob
import cv2
import torch
import numpy as np
from functools import partial
from utils.datasets import letterbox
from torch.utils.data import Dataset

def transform_yolo(img0):
    img = letterbox(img0, new_shape=max(img0.shape))[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    img /= 255.0
    return img

def transform_effdet(img0, image_size=640):
    # TODO: try various resize methods
    img = cv2.resize(img0, (image_size, image_size))
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).float()
    img /= 255.0
    return img


class Vehicle(Dataset):
    def __init__(self, images_root, model='yolo', image_size_effdet=640):
        super().__init__()
        self.files = sorted(glob.glob(os.path.join(images_root, '*jpg')))
        if len(self.files) == 0:
            self.files = sorted(glob.glob(os.path.join(images_root, '*png')))
        
        assert len(self.files) > 0, 'No images found'

        if 'yolo' in model:
            self.transform = partial(transform_yolo)
        elif 'effdet' in model:
            self.transform = partial(transform_effdet, image_size=image_size_effdet)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, frame_id):
        img0 = cv2.imread(self.files[frame_id])
        img = self.transform(img0)
        return img, img0, frame_id + 1



