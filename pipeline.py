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

class Pipeline:
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

        self.tracker = JDETracker(max_age=args.max_age, buffer_size=args.buffer_size, det_thresh=args.conf_thres,
                                  thresh1=args.tracker_thresh1, thresh2=args.tracker_thresh2, thresh3=args.tracker_thresh3)
        self.track_features_type = args.track_features_type
        
        self.save_detects = args.save_detects
        self.detect_outputs = args.detect_outputs
        self.detect_results = []
        
        self.track_outputs = args.track_outputs
        self.track_results = []

    def run(self):
        pbar = tqdm(self.loader)
        
        i = 0
        for imgs, imgs0, frame_ids in pbar:
            preds = self.detector(imgs, imgs0)
            
            i += 1            
            if self.args.test_docker and i == 5:
                break
            
            for xi, det in enumerate(preds):
                if det is None:
                    continue
                det = det.cpu().numpy()
            
                img = imgs0[xi].numpy()
                features = compute_track_features(self.track_features_type, det, img)                
                online_targets = self.tracker.update(det, features=features)
                if len(online_targets) > 0:
                    tracks = [np.r_[int(frame_ids[xi]), t.track_id, t.tlwh, t.score, t.label] for t in online_targets]
                    tracks = np.array(tracks)
                    self.track_results.append(tracks)
                
                # concate features
                det = np.c_[det[:, :6], features]                
                frames = np.ones((det.shape[0], 1), dtype=np.int) * int(frame_ids[xi])
                res = np.c_[frames, det]
                self.detect_results.append(res) 
    
    def save(self):
        # consider save this one, because too large
        if self.save_detects:
            self.detect_results = np.concatenate(self.detect_results, axis=0)
            features = [f"f{i}" for i in range(self.detect_results[:, 7:].shape[1])]
            columns = ['frame_id', 'x1', 'y1', 'x2', 'y2', 'score', 'label']
            columns.extend(features)
            df = pd.DataFrame(self.detect_results, columns=columns)
            df.to_csv(self.detect_outputs, index=False, sep= ' ')
        
        self.track_results = np.concatenate(self.track_results, axis=0)
        columns = ['frame_id', 'track_id', 'x', 'y', 'w', 'h', 'score', 'label']
        df = pd.DataFrame(self.track_results, columns=columns)
        df.to_csv(self.track_outputs, index=False, sep=' ')
