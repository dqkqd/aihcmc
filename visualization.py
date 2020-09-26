import os 
import glob
import argparse
import random
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from utils.interest import get_roi, get_smaller_roi

FPS = 10

def run(args):
    roi, height, width = get_roi(args.annofile)
    roi_region = np.r_[roi, roi[0].reshape(1, 2)].round().astype(np.int)

    if args.use_smaller_roi:
        smaller_roi = get_smaller_roi(roi, args.smaller_roi_alpha)
        smaller_roi_region = np.r_[smaller_roi, smaller_roi[0].reshape(1, 2)].round().astype(np.int)
    
    classes = ["1", "2", "3", "4"]
    roi_color = [0, 0, 255]
    smaller_roi_color = [0, 0, 0]

    classes_color = [
        (0, 255, 0),
        (255, 0, 0),
        (255, 255, 0),
        (0, 255, 255),
    ]

    fourcc = "mp4v"
    vid_writer = cv2.VideoWriter(args.outputvideo, cv2.VideoWriter_fourcc(*fourcc), FPS, (width, height))
    images = sorted(glob.glob(os.path.join(args.images_root, args.videoname, "*")))
    track_results = pd.read_csv(args.track_outputs, sep=' ')
    track_results['x2'] = track_results['x'] + track_results['w']
    track_results['y2'] = track_results['y'] + track_results['h']

    frame_id = 0
    for img_path in tqdm(images):
        frame_id += 1
        img = cv2.imread(img_path)
        
        record = track_results[track_results['frame_id']==float(frame_id)]
        record = record[record['alive']==1]
        
        for i in range(len(roi_region) - 1):
            start = tuple(roi_region[i])
            end = tuple(roi_region[i+1])
            cv2.line(img, start, end, color=roi_color, thickness=2)

            if args.use_smaller_roi:
                start = tuple(smaller_roi_region[i])
                end = tuple(smaller_roi_region[i+1])
                cv2.line(img, start, end, color=smaller_roi_color, thickness=2)
        
        if len(record) > 0:
            track_ids = record['track_id'].values.astype(np.int)
            boxes = record[['x', 'y', 'x2', 'y2']].values.astype(np.int)
            scores = record['score'].values
            labels = record['label'].values.astype(int)

            for track_id, box, score, label in zip(track_ids, boxes, scores, labels):
                color = classes_color[label - 1]
                name = classes[label - 1]
                box = box.astype(np.int)

                # draw box
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), color=color, thickness=2)

                # draw id
                cv2.putText(img, f"id:{track_id}", (box[0] - 1, box[1] - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, color=color, thickness=2)
            
                # draw class name
                cv2.putText(img, f"{name}:{score:.2f}", (box[0] + 1, box[3] + 20),
                    cv2.FONT_HERSHEY_COMPLEX, 0.6, color=color, thickness=2)
        
        vid_writer.write(img)
    

    vid_writer.release()
        

        
        

    


