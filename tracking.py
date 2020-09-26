import pandas as pd
import numpy as np
from tracker.multitracker import JDETracker



class Tracking:
    def __init__(self, args):
        self.args = args
        self.tracker = JDETracker(max_age=args.max_age, buffer_size=args.buffer_size, det_thresh=args.conf_thres,
                                  thresh1=args.tracker_thresh1, thresh2=args.tracker_thresh2, thresh3=args.tracker_thresh3)
        
        self.track_outputs = args.track_outputs
        self.track_results = []
        
        
    def run(self):
        df = pd.read_csv(self.args.detect_outputs, sep=' ')
        frames = sorted(df['frame_id'].unique())
        for frame_id in frames:
            frame_id = int(frame_id)
            records = df[df['frame_id']==frame_id]
            dets = records.values
            dets = dets[:, 1:]
            # det: columns = ['x1', 'y1', 'x2', 'y2', 'score', 'label', features]
            # features = dets[:, 6:]
            online_targets = self.tracker.update(dets[:, :6], dets[:, 6:])
            if len(online_targets) == 0:
                continue 
            tracks = [np.r_[frame_id, t.track_id, t.tlwh, t.score, t.label] for t in online_targets]
            tracks = np.array(tracks)
            self.track_results.append(tracks)
    
    def save(self):
        # must save
        self.track_results = np.concatenate(self.track_results, axis=0)
        columns = ['frame_id', 'track_id', 'x', 'y', 'w', 'h', 'score', 'label']
        df = pd.DataFrame(self.track_results, columns=columns)
        df.to_csv(self.track_outputs, index=False, sep=' ')
            
            
        
        