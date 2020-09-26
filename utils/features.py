import numpy as np
import cv2


def compute_track_features(track_features_type, det, img=None):

    if track_features_type == 'label':
        features = det[:, 6:]
        assert features.shape[1] == 4

    elif track_features_type == 'label_color_hist':
        features = np.zeros((det.shape[0], 256 * 3 * 4))
        for i, box in enumerate(det):
            box = box.astype(np.int)
            small_img = img[box[1] : box[3], box[0] : box[2]]
            rc = cv2.calcHist([small_img], [0], None, [256], [0, 256])
            gc = cv2.calcHist([small_img], [1], None, [256], [0, 256])
            bc = cv2.calcHist([small_img], [2], None, [256], [0, 256])
            feature = np.r_[rc, gc, bc].ravel()
            features[i][3 * 256 * (box[5] - 1) : 3 * 256 * (box[5])] = feature
    
    else:
        features = det[:, 5].reshape(-1, 1)

    return features 
        
