import os 
import json
import argparse
from tqdm import tqdm
import movement
import visualization
import warnings
warnings.filterwarnings('ignore')

import random
import numpy as np
import torch

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test_docker', action='store_true')
    
    parser.add_argument('--detecting', action='store_true')
    parser.add_argument('--tracking', action='store_true')
    parser.add_argument('--counting', action='store_true')
    
    parser.add_argument('--seed', type=int, default=2020)
    parser.add_argument('--videoname', type=str, default='')
    parser.add_argument('--images-root', type=str, default='images_root')
    parser.add_argument('--anno-path', type=str, default='test_data')

    # dataloader
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=4)

    # detector
    parser.add_argument('--weight', type=str, default='effdet-d5-640.pt')
    parser.add_argument('--model-type', type=str, default='effdet5')
    parser.add_argument('--image-size-effdet', type=int, default=640) #only works with effdet
    parser.add_argument('--conf-thres', type=float, default=0.45)
    parser.add_argument('--iou-thres', type=float, default=0.5)
    parser.add_argument('--detect-outputs-root', type=str, default='detect_outputs')
    parser.add_argument('--save-detects', action='store_true')
    
    # tracker
    parser.add_argument('--fps', type=int, default=10)
    parser.add_argument('--max-age', type=int, default=2)
    parser.add_argument('--buffer-size', type=int, default=30)
    parser.add_argument('--track-features-type', type=str, default='label', help="label, label_color_hist")
    parser.add_argument('--tracker-thresh1', type=float, default=0.38)
    parser.add_argument('--tracker-thresh2', type=float, default=0.91)
    parser.add_argument('--tracker-thresh3', type=float, default=0.38)
    parser.add_argument('--track-outputs-root', type=str, default='track_outputs')


    # counting
    parser.add_argument('--count-params-root', type=str, default='counting_params_root')
    parser.add_argument('--count-outputs-root', type=str, default='counting_outputs')
    parser.add_argument('--heat-map-root', type=str, default='heatmaps')
    parser.add_argument('--nframes', type=int, default=13500)
    
    # finetuning
    parser.add_argument('--tuning', action='store_true')
    parser.add_argument('--K', type=int, default=4)
    parser.add_argument('--tuning-submission', type=str, default='tuning_submission.txt')
    
    # visualization
    parser.add_argument('--visualization', action='store_true')
    parser.add_argument('--videos-outputs-root', type=str, default='output_videos')


    args = parser.parse_args()
    
    args.annofile = os.path.join(args.anno_path, args.videoname + ".json")
    args.detect_outputs = os.path.join(args.detect_outputs_root, args.videoname + ".txt")
    args.track_outputs = os.path.join(args.track_outputs_root, args.videoname + ".txt")
    args.count_outputs = os.path.join(args.count_outputs_root, args.videoname + ".txt")
    args.outputvideo = os.path.join(args.videos_outputs_root, args.videoname + ".mp4")
    args.count_params = os.path.join(args.count_params_root, args.videoname + ".json")
    
    
    os.makedirs(args.detect_outputs_root, exist_ok=True)
    os.makedirs(args.track_outputs_root, exist_ok=True)
    os.makedirs(args.videos_outputs_root, exist_ok=True)
    os.makedirs(args.count_params_root, exist_ok=True)
    os.makedirs(args.count_outputs_root, exist_ok=True)
    
    images_folder = os.path.join(args.images_root, args.videoname)
    args.nframes = len(os.listdir(images_folder))
    
    seed_everything(args.seed)
    
    # if tuning, need to save_detects
    if args.tuning:
        args.save_detects = True
        
    # detect and tracking
    if args.detecting and args.tracking:
        from pipeline import Pipeline
        try:
            detect_and_track = Pipeline(args)
            detect_and_track.run()
            detect_and_track.save()
        except:
            args.batch_size = 8
            detect_and_track = Pipeline(args)
            detect_and_track.run()
            detect_and_track.save()
            
            
    
    # tracking only
    if args.tracking and not args.detecting:
        from tracking import Tracking
        tracking = Tracking(args)
        tracking.run()
        tracking.save()
    # No need to run detect without tracking because the speed is the same as detect_and_track
        
    # counting
    if args.counting:
        import counting
        if os.path.isfile(args.count_params):
            with open(args.count_params, 'r') as f:
                cam_args = json.load(f)
        else:
            cam_args = counting.DEFAULT_ARGS
        counter = counting.Movement(args, **cam_args)
        counter.run()
        counter.save()
        
    if False: #args.tuning:
        from evaluation import eval_video
        diff, scale = eval_video(args)
        string = f'{args.videoname}, diff = {diff}, scale = {scale}'
        print(string)
        my_file = 'check.txt'        
        with open(my_file, 'a+') as f:
            f.write(string + "\n")
    
    # # classify movement
    # movement.run(args)


    # # visualization
    # if args.visualization:
    #     visualization.run(args)