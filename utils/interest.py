import json
import numpy as np
from numba import jit
from PIL import Image, ImageDraw

def get_roi(annofile):
    with open(annofile, 'r') as f:
        info = json.load(f)
    
    height, width = info["imageHeight"], info["imageWidth"]

    data = info['shapes'][0]
    assert data['label'] == 'zone', 'zone movement missing'
    roi = np.array(data['points'])

    return roi, height, width

def get_moi(annofile):
    with open(annofile, 'r') as f:
        info = json.load(f)
    moi = {}
    for data in info['shapes']:
        if data['label'] != 'zone':
            direction = int(data['label'].replace('direction', ''))
            moi[direction] = np.array(data['points'])
    return moi

def get_smaller_roi(roi, alpha, beta=1):
    center_of_mass = roi.sum(axis=0) / roi.shape[0]
    smaller_roi = (roi * alpha + center_of_mass * beta) / (alpha + beta)
    return smaller_roi


def get_mask(width, height, roi):
    img = Image.new('L', (width, height), 0)
    region = roi.flatten().tolist()
    ImageDraw.Draw(img).polygon(region, outline=0, fill=255)
    return np.array(img).astype(np.bool)

@jit
def filter_outside_mask(det, mask):
    height, width = mask.shape
    # det is xyxy
    det[:, 0:2] = np.floor(det[:, 0:2]).astype(np.int)
    det[:, 2:4] = np.floor(det[:, 2:4]).astype(np.int)
    det[:, 0:4:2] = np.clip(det[:, 0:4:2], 0, width - 1)
    det[:, 1:4:2] = np.clip(det[:, 1:4:2], 0, height - 1)

    xs_int = ((det[:, 0] + det[:, 2]) // 2).astype(np.int)
    ys_int = ((det[:, 1] + det[:, 3]) // 2).astype(np.int)

    indices = []

    for i, row in enumerate(det[:, :4]):
        row = row.astype(np.int)
        if not mask[ys_int[i], xs_int[i]]:
            bbox_size = (row[2] - row[0]) * (row[3] - row[1])
            overlap_size = mask[row[1] : row[3], row[0]: row[2]].sum()
            overlap_frac = overlap_size / bbox_size
            if overlap_frac == 0:
                continue
        indices.append(i)
    
    return indices


