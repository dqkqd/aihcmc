import os
import cv2
import time
import numpy as np
import pandas as pd
import torch
from numba import jit
from tqdm import tqdm
from dataset.dataset import Vehicle
from detector.detector import Detector
from tracker.multitracker import JDETracker
from utils import interest
from utils.features import compute_track_features
from torch.utils.data import DataLoader, SequentialSampler

class Detect:
    def __init__(self, args):

        self.args = args
        self.num_classes = 4

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert self.device == 'cuda', 'CUDA is not available'

        images_folder = os.path.join(args.images_root, args.videoname)

        self.dataset = Vehicle(images_folder, model=args.model_type, image_size_effdet=args.image_size_effdet)
        self.loader = DataLoader(
            self.dataset, batch_size=args.batch_size, shuffle=False,
            sampler=SequentialSampler(self.dataset),
            num_workers=args.num_workers, pin_memory=False, drop_last=False)

        self.detector = Detector(args.weight, args.conf_thres, args.iou_thres, args.model_type, args.image_size_effdet)

        self.detect_results = []
        self.detect_outputs = args.detect_outputs

    def run(self):
        pbar = tqdm(self.loader)
        for imgs, imgs0, frame_ids in pbar:
            preds = self.detector(imgs, imgs0)
            for xi, det in enumerate(preds):
                if det is None:
                    continue
                det = det.cpu().numpy()
                frames = np.ones((det.shape[0], 1), dtype=np.int) * int(frame_ids[xi])
                res = np.c_[frames, det]
                self.detect_results.append(res)
        self.detect_results = np.concatenate(self.detect_results, axis=0)
        df = pd.DataFrame(self.detect_results, columns=['frame_id', 'x1', 'y1', 'x2', 'y2', 'score',
                                                        'label', 'f0', 'f1', 'f2', 'f3'])
        df.to_csv(self.detect_outputs, index=False, sep= ' ')
        