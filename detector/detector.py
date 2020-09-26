import os
import cv2
import torch
import torchvision
import numpy as np
from .effdet import get_efficientdet_config, EfficientDet, DetBenchTrain, DetBenchPredict
from .effdet.efficientdet import HeadNet

from utils.general import scale_coords, xywh2xyxy
from utils.datasets import letterbox

if not os.path.isdir('models'):
    os.symlink('detector/models', 'models')

class Effdet:
    def __init__(self,
        weight,
        model_type,
        image_size=640,
        conf_thres=0.4,
        iou_thres=0.5):

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert self.device == 'cuda'
        
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.image_size = image_size
        self.num_classes = 4

        level = int(model_type[-1])
        assert level in [3, 4, 5]
        config = get_efficientdet_config(f'tf_efficientdet_d{level}')
        self.model = EfficientDet(config, pretrained_backbone=False)
        config.num_classes = self.num_classes
        config.image_size = self.image_size
        self.model.class_net = HeadNet(config, num_outputs=config.num_classes, norm_kwargs=dict(eps=.001, momentum=.01))
        checkpoint = torch.load(weight) #
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = DetBenchPredict(self.model, config)
        self.model.eval()
        self.model.to(self.device)

    
    @torch.no_grad()
    def __call__(self, imgs, imgs0):
        imgs = imgs.to(self.device)        
        img_size = torch.tensor([imgs[0].shape[-2:]] * imgs.shape[0]).float().to(self.device)
        img0_size = torch.tensor(imgs0[0].shape).float().to(self.device)
        preds = self.model(imgs, img_size, img0_size, self.conf_thres, self.iou_thres)
        return preds

class YOLO:
    def __init__(self, weight, conf_thres=0.4, iou_thres=0.5):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert self.device == 'cuda', "cuda is not available"
        checkpoint = torch.load(weight, map_location=self.device)
        self.model = checkpoint['model'].float().fuse().to(self.device)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # TODO: consider change this
        self.augment = False
   
    @torch.no_grad()
    def __call__(self, imgs, imgs0):
        imgs = imgs.to(self.device)
        preds = self.model(imgs, augment=self.augment)[0]
        preds = non_max_suppression(preds, imgs[0].shape[-2:], imgs0[0].shape,
            conf_thres=self.conf_thres, iou_thres=self.iou_thres)
        return preds


class Detector:
    def __init__(self,
        weight,
        conf_thres=0.4,
        iou_thres=0.5,
        model_type='yolo',
        image_size_effdet=640, #only works with effdet
        ):

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        self.model_type = model_type
        if 'effdet' in self.model_type:
            self.model = Effdet(weight, self.model_type, image_size_effdet, conf_thres, iou_thres)
        elif 'yolo' in self.model_type:
            self.model = YOLO(weight, conf_thres, iou_thres)
    
    def __call__(self, imgs, imgs0):
        return self.model(imgs, imgs0)

def non_max_suppression(prediction, img_shape, img0_shape, conf_thres=0.1, iou_thres=0.6, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = 4  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float(), x[i, 5:]), 1)

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores

        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        x = x[i]

        x[:, :4] = scale_coords(img_shape, x[:, :4], img0_shape).round()
        # label needs start from 1
        x[:, 5] += 1

        output[xi] = x

    return output
