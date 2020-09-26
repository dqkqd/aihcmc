import os
import argparse
import numpy as np
import pandas as pd
from numba import jit
from tqdm import tqdm
from utils.interest import get_moi, get_roi, get_mask, get_smaller_roi
from scipy.spatial.distance import directed_hausdorff, cosine 
import warnings
warnings.filterwarnings('ignore')

def find_moi_hausdorff(movement, moi, use_symmetry=False, use_vector_direction=False):
    moi_id = -1
    min_moi = np.inf

    for direction, moi_arr in moi.items():
        if use_vector_direction:
            v1 = movement[-1] - movement[0]
            v2 = moi_arr[-1] - moi_arr[0]
            cos = 1 - cosine(v1, v2)
            theta = np.arccos(cos)
            if theta > np.pi / 2:
                continue

        moi_res = directed_hausdorff(movement, moi_arr)[0]
        if use_symmetry:
            moi_res = max(moi, directed_hausdorff(moi_arr, movement)[0])
        if moi_res < min_moi:
            min_moi = moi_res
            moi_id = direction

    return moi_id

@jit
def tlwh2xcyc(boxes):
    xc = boxes[:, 0] + boxes[:, 2] / 2
    yc = boxes[:, 1] + boxes[:, 3] / 2
    xcyc = np.c_[xc, yc]
    return xcyc

def run(args):
    
    moi = get_moi(args.annofile)

    roi, height, width = get_roi(args.annofile)
    # mask = get_mask(width, height, roi)
    if args.use_smaller_roi:
        smaller_roi = get_smaller_roi(roi, args.smaller_roi_alpha)
        smaller_mask = get_mask(width, height, smaller_roi)
    
    videoname = args.videoname.split('.')[0]

    track_results = pd.read_csv(args.track_outputs, sep=' ')

    track_ids = track_results['track_id'].unique()

    results = []

    track_results['alive'] = 0
    
    for track_id in tqdm(track_ids):
        record = track_results[track_results['track_id'] == track_id]
        frames = record['frame_id'].values
        label = record['label'].values.mean().round()

        if frames[-1] - frames[0] < args.min_length:
            continue
        
        if len(frames) < args.min_frames:
            continue

        boxes = record[['x', 'y', 'w', 'h']].values
        movement = tlwh2xcyc(boxes)

        if args.use_smaller_roi:
            xc_int = int(movement[-1][0])
            yc_int = int(movement[-1][1])
            if smaller_mask[yc_int, xc_int]:
                continue
        
        if args.use_hausdorff:
            moi_id = find_moi_hausdorff(movement, moi,
                use_symmetry=args.use_symmetry,
                use_vector_direction=args.use_vector_direction)
        else:
            raise NotImplementedError

        if moi_id == -1:
            continue

        item = [videoname, int(frames[-1]), int(moi_id), int(label)]
        results.append(item)
        results = sorted(results, key=lambda x: x[1])

        if args.visualization:
            track_results.loc[track_results['track_id'] == track_id, ['alive']] = 1

    
    pd.DataFrame(np.array(results)).to_csv(args.trajectory_outputs, sep=' ', index=False, header=False)
    if args.visualization:
        track_results.to_csv(args.track_outputs, sep=' ', index=False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--videoname', type=str, default='')
    parser.add_argument('--annofile', type=str, default='')
    parser.add_argument('--track_outputs', type=str, default='')
    parser.add_argument('--trajectory_outputs', type=str)

    # filter 
    parser.add_argument('--min_length', type=int, default=3)

    # hausdorff
    parser.add_argument('--use_hausdorff', action='store_true')
    parser.add_argument('--use_symmetry', action='store_true')
    parser.add_argument('--use_vector_direction', action='store_true')

    # save track_preprocessed for visualization
    parser.add_argument('--visualization', action='store_true')

    args = parser.parse_args()

    run(args)


