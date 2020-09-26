import os
import json
import glob
import math
import pickle
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from collections import namedtuple, defaultdict
from enum import IntEnum, auto
from scipy.ndimage import gaussian_filter1d
from scipy.special import expit
from scipy.stats import linregress, norm
from tqdm import tqdm
from utils.interest import get_roi, get_mask, get_moi

TrackItem = namedtuple('TrackItem', ['frame_id', 'obj_type', 'data'])
DEFAULT_ARGS = {
    "fps": 10,
    "stride": 1,
    "min_score": 0.1,
    "overlap_thres": 0.1,
    "gaussian_std": 0.3,
    "min_length": 0.3,
    "speed_window": 1,
    "min_speed": 10,
    "distance_slope_scale": 2,
    "proportion_scale": 0.8,
    "proportion_thres_to_delta": 0.5,
    "start_period": 0.3,
    "start_thres": 0.5,
    "start_proportion_factor": 1.5,
    "plus_frame1": list(range(0, 12)),
    "plus_frame2": list(range(0, 12)),
    "merge_detection_score": False,
    "final": True,
    "use_score1": False,
    "distance_scale": 5,
    "distance_base_size": 4,
}
class ObjectType(IntEnum):
    '''
        Loại 1: xe 2 bánh như xe đạp, xe máy
        Loại 2: xe 4-7 chỗ như xe hơi, taxi, xe bán tải…
        Loại 3: xe trên 7 chỗ như xe buýt, xe khách
        Loại 4: xe tải, container, xe cứu hỏa
    '''
    Type1 = 1
    Type2 = 2
    Type3 = 3
    Type4 = 4

def get_movement_heatmaps(movements, height, width):
    distance_heatmaps = np.empty((len(movements), height, width))
    proportion_heatmaps = np.empty((len(movements), height, width))
    xs, ys = np.meshgrid(np.arange(width), np.arange(height))
    points = np.stack([xs.flatten(), ys.flatten()], axis=1)
    for label, movement_vertices in movements.items():
        vectors = movement_vertices[1:] - movement_vertices[:-1]
        lengths = np.linalg.norm(vectors, axis=-1) + 1e-4
        rel_lengths = lengths / lengths.sum()
        vertex_proportions = np.cumsum(rel_lengths)
        vertex_proportions = np.concatenate([[0], vertex_proportions[:-1]])
        offsets = ((points[:, None] - movement_vertices[None, :-1])
                   * vectors[None]).sum(axis=2)
        fractions = np.clip(offsets / (lengths ** 2), 0, 1)
        targets = movement_vertices[:-1] + fractions[:, :, None] * vectors
        distances = np.linalg.norm(points[:, None] - targets, axis=2)
        nearest_segment_ids = distances.argmin(axis=1)
        nearest_segment_fractions = fractions[
            np.arange(fractions.shape[0]), nearest_segment_ids]
        distance_heatmap = distances.min(axis=1)
        proportion_heatmap = vertex_proportions[nearest_segment_ids] + \
            rel_lengths[nearest_segment_ids] * nearest_segment_fractions
        distance_heatmaps[label - 1, ys, xs] = distance_heatmap.reshape(
            height, width)
        proportion_heatmaps[label - 1, ys, xs] = proportion_heatmap.reshape(
            height, width)
    return distance_heatmaps, proportion_heatmaps

class Movement:
    def __init__(self, args, fps=10, stride=1,
                 overlap_thres=0.1, gaussian_std=0.3, min_length=0.3, speed_window=1,
                 min_speed=10, distance_scale=5, distance_base_size=4, distance_slope_scale=2,
                 proportion_thres_to_delta=0.5, proportion_scale=0.8, 
                 start_period=0.3, start_thres=0.5, start_proportion_factor=1.5,
                 merge_detection_score=False, final=True, use_score1=False, min_score=0.1,
                 plus_frame1=list(range(12)), plus_frame2=list(range(12))):
                
        self.args = args
                   
        self.videoname = self.args.videoname
        self.track_outputs = self.args.track_outputs
        self.annofile = self.args.annofile
        
        self.region, self.height, self.width = get_roi(self.annofile)
        self.region_mask = get_mask(self.width, self.height, self.region)
        self.movements = get_moi(self.annofile)
        print(f"Cam has {len(self.movements)} moi")
        
        #####################################################################
        heatmap_dir = self.args.heat_map_root
        if not os.path.isdir(heatmap_dir):
            os.makedirs(heatmap_dir)
        heatmap_file = os.path.join(heatmap_dir, self.videoname + ".pkl")
        if not os.path.isfile(heatmap_file):
            self.distance_heatmaps, self.proportion_heatmaps = \
                get_movement_heatmaps(self.movements, self.height, self.width)
            with open(heatmap_file, 'wb') as f:
                pickle.dump([self.distance_heatmaps, self.proportion_heatmaps], f)
        else:
            with open(heatmap_file, 'rb') as f:
                self.distance_heatmaps, self.proportion_heatmaps = pickle.load(f)
        #####################################################################
        
        
        self.fps = fps
        self.stride = stride
        self.overlap_thres = overlap_thres
        self.gaussian_std = gaussian_std
        if gaussian_std is not None:
            self.gaussian_std = gaussian_std * fps
        self.min_length = max(3, min_length * fps)
        self.speed_window = int(speed_window * fps // 2) * 2
        self.min_speed = min_speed / fps * self.speed_window
        self.distance_scale = distance_scale
        self.distance_base_size = distance_base_size
        self.distance_slope_scale = distance_slope_scale
        self.proportion_thres_to_delta = proportion_thres_to_delta,
        self.proportion_factor = 1 / proportion_scale
        self.start_period = start_period * fps
        self.start_thres = start_thres
        self.start_proportion_factor = start_proportion_factor
        self.merge_detection_score = merge_detection_score
        self.final = final
        self.use_score1 = use_score1
        self.min_score = min_score
        self.plus_frame1 = plus_frame1
        self.plus_frame2 = plus_frame2
        
        self.track_items = defaultdict(list)
        
        self.count_results = []
        self.count_outputs = self.args.count_outputs
        
        
        
    def forward_tracks(self):
        df = pd.read_csv(self.track_outputs, sep=' ')    
        
        boxes = df[['x', 'y', 'w', 'h']].values
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
        
        locations = ((boxes[:, :2] + boxes[:, 2:]) / 2)
        locations[:, 0] = np.clip(locations[:, 0], 0, self.width - 1)
        locations[:, 1] = np.clip(locations[:, 1], 0, self.height - 1)
        diagonals = np.linalg.norm(boxes[:, 2:] - boxes[:, :2], axis=1)
        boxes[:, ::2] = np.clip(boxes[:, ::2], 0, self.width)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, self.height)
        
        xy0s = np.floor(boxes[:, :2]).astype(np.int)
        xy1s = np.ceil(boxes[:, 2:]).astype(np.int)
        
        df['xc'] = locations[:, 0]
        df['yc'] = locations[:, 1]
        df['diag'] = diagonals
        df['x0'] = xy0s[:, 0]
        df['y0'] = xy0s[:, 1]
        df['x1'] = xy1s[:, 0]
        df['y1'] = xy1s[:, 1]
        
        track_ids = sorted(df['track_id'].unique())
        
        for track_id in track_ids:
            records = df[df['track_id']==track_id]
            frames_id = records['frame_id'].values.astype(np.int)
            obj_types = records['label'].values.astype(np.int)
            scores = records['score'].values
            diagonals = records['diag'].values
            locations = records[['xc', 'yc']].values
            boxes = records[['x0', 'y0', 'x1', 'y1']].values                        
            for i in range(len(records)):
                x , y = locations[i]
                x_int = x.round().astype(np.int)
                y_int = y.round().astype(np.int)
                if not self.region_mask[y_int, x_int]:
                    x0, y0, x1, y1 = boxes[i]
                    box_size = (y1 - y0) * (x1 - x0)
                    overlap_size = self.region_mask[y0:y1, x0:x1].sum()
                    overlap_frac = overlap_size / box_size
                    if overlap_frac <= self.overlap_thres:
                        continue
                item = TrackItem(frames_id[i], obj_types[i], (x, y, diagonals[i], scores[i]))
                self.track_items[track_id].append(item)
    

    def get_obj_type(self, track_items, track):        
        active_frame_ids = set(track[:, -1].tolist())
        obj_types = [t.obj_type for t in track_items
                     if t.frame_id in active_frame_ids]    
        type_counts = np.bincount(obj_types)
        
        if len(type_counts) > 0:
            obj_type = type_counts.argmax()            
            assert obj_type == track_items[0].obj_type
        else:
            obj_type = track_items[0].obj_type
        
        return obj_type
        
    
    # get track of a track id
    def get_track(self, track_items):
        init_frame_id = track_items[0].frame_id
        length = track_items[-1].frame_id - init_frame_id + 1
        
        if length < self.min_length:
            return None
        
        if len(track_items) == length:
            interpolated_track = np.stack([t.data for t in track_items])
        else:
            interpolated_track = np.empty((length, len(track_items[0].data)))
            interpolated_track[:, 0] = -1
            for t in track_items:
                interpolated_track[t.frame_id - init_frame_id] = t.data
            for frame_i, state in enumerate(interpolated_track):
                if state[0] >= 0:
                    continue
                for left in range(frame_i - 1, -1, -1):
                    if interpolated_track[left, 0] >= 0:
                        left_state = interpolated_track[left]
                        break
                for right in range(frame_i + 1, interpolated_track.shape[0]):
                    if interpolated_track[right, 0] >= 0:
                        right_state = interpolated_track[right]
                        break
                try:
                    movement = right_state - left_state
                except:
                    continue
                ratio = (frame_i - left) / (right - left)
                interpolated_track[frame_i] = left_state + ratio * movement
                
        if self.gaussian_std is not None:
            track = gaussian_filter1d(
                interpolated_track, self.gaussian_std, axis=0, mode='nearest')
        else:
            track = interpolated_track
        track = np.hstack([track, np.arange(
            init_frame_id, init_frame_id + length)[:, None]])
        speed_window = min(self.speed_window, track.shape[0] - 1)
        speed_window_half = speed_window // 2
        speed_window = speed_window_half * 2
        speed = np.linalg.norm(
            track[speed_window:, :2] - track[:-speed_window, :2], axis=1)
        speed_mask = np.zeros((track.shape[0]), dtype=np.bool)
        speed_mask[speed_window_half:-speed_window_half] = \
            speed >= self.min_speed
        speed_mask[:speed_window_half] = speed_mask[speed_window_half]
        speed_mask[-speed_window_half:] = speed_mask[-speed_window_half - 1]
        track = track[speed_mask]
        track_int = track[:, :2].round().astype(int)
        iou_mask = self.region_mask[track_int[:, 1], track_int[:, 0]]
        track = track[iou_mask]
        if track.shape[0] < self.min_length:
            return None
        return track
        
        
    def get_movement_scores(self, track, obj_type):
        positions = track[:, :2].round().astype(int)
        diagonals = track[:, 2]
        detection_scores = track[:, 3]
        frame_ids = track[:, -1]
        distances = self.distance_heatmaps[:, positions[:, 1], positions[:, 0]]
        proportions = self.proportion_heatmaps[
            :, positions[:, 1], positions[:, 0]]
        distances = distances / diagonals[None]
        mean_distances = distances.mean(axis=1)
        x = np.linspace(0, 1, proportions.shape[1])
        distance_slopes = np.empty((len(self.movements)))
        proportion_slopes = np.empty((len(self.movements)))
        for movement_i in range(len(self.movements)):
            distance_slopes[movement_i] = linregress(
                x, distances[movement_i])[0]
            proportion_slopes[movement_i] = linregress(
                x, proportions[movement_i])[0]
        proportion_delta = proportions.max(axis=1) - proportions.min(axis=1)
        proportion_slopes = np.where(
            proportion_slopes >= self.proportion_thres_to_delta,
            proportion_delta, proportion_slopes)
        
        if self.use_score1:
            if obj_type == ObjectType.Type4:
                distance_base_scale = min(
                    1, self.distance_base_size / mean_distances.shape[0])
                distance_base = np.sort(mean_distances)[
                    :self.distance_base_size].sum() * distance_base_scale
                score_1 = 1 - (mean_distances / distance_base) ** 2
            else:
                score_1 = expit(4 - mean_distances * self.distance_scale)
            
        score_2 = self.proportion_factor * np.minimum(
            proportion_slopes, 1 / (proportion_slopes + 1e-8))
        if frame_ids[0] <= self.start_period and \
                score_2.max() <= self.start_thres:
            score_2 *= self.start_proportion_factor
        score_3 = norm.pdf(distance_slopes * self.distance_slope_scale) / 0.4
        
        if self.use_score1:
            scores = np.stack([score_1, score_2, score_3], axis=1)
        else:
            scores = np.stack([score_2, score_3], axis=1)
            
        if self.final:
            scores = np.clip(scores, 0, 1).prod(axis=1)
            if self.merge_detection_score:
                scores = scores * detection_scores.mean()
        return scores
    

    def run(self):
        self.forward_tracks()
        for track_items in self.track_items.values():
            track = self.get_track(track_items)
            if track is None:
                continue
            obj_type = self.get_obj_type(track_items, track)
            movement_scores = self.get_movement_scores(track, obj_type)
            max_index = movement_scores.argmax()
            max_score = movement_scores[max_index]
            if max_score < self.min_score:
                continue
            frame_id = track_items[-1][0] + 1
            
            if obj_type == 1:
                frame_id += self.plus_frame1[max_index]
            else:
                frame_id += self.plus_frame2[max_index]
            
            if frame_id > self.args.nframes:
                continue
            if frame_id < 0:
                continue
            track_res = [self.videoname, frame_id, max_index + 1, obj_type]
            self.count_results.append(track_res)
        
        # sort by frame_id
        self.count_results = sorted(self.count_results, key=lambda x: x[1])
            
    def save(self):
        with open(self.count_outputs, 'w') as f:
            for res in self.count_results:
                line = list(map(str, res))
                line = ' '.join(line) + "\n"
                f.write(line)