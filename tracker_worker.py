import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal

from darknet import Darknet
from utils import (
    objectness_filter_and_nms,
    scale_numbers,
    letterbox_pad,
    get_corner_coords,
)


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def cx(self):
        return (self.x1 + self.x2) // 2

    @property
    def cy(self):
        return (self.y1 + self.y2) // 2

    @property
    def w(self):
        return self.x2 - self.x1

    @property
    def h(self):
        return self.y2 - self.y1

    def to_csrt_rect(self):
        """Returns (x, y, w, h) tuple for OpenCV tracker init."""
        return (self.x1, self.y1, self.w, self.h)


@dataclass
class FrameResult:
    frame: np.ndarray
    detections: List
    tracked_rect: Optional[Tuple[int, int, int, int]]
    tracked_class: Optional[str]
    mode: str
    fps: float


def find_best_matching_detection(
    tracker_rect: Tuple[int, int, int, int],
    detections: List,
    min_iou: float = 0.3,
) -> Optional['Detection']:
    """Return the detection with highest IoU vs tracker_rect. None if best < min_iou."""
    if not detections:
        return None

    tx, ty, tw, th = tracker_rect
    tx2, ty2 = tx + tw, ty + th
    best_det = None
    best_iou = min_iou

    for det in detections:
        ix1 = max(tx, det.x1)
        iy1 = max(ty, det.y1)
        ix2 = min(tx2, det.x2)
        iy2 = min(ty2, det.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter == 0:
            continue
        union = tw * th + det.w * det.h - inter
        iou = inter / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_det = det

    return best_det


class InferenceEngine:
    """Wraps YOLOv3. Call detect() from one thread at a time."""

    def __init__(self, weights_path, config_path, labels_path, device=None):
        self.device = device or torch.device('cpu')
        with open(labels_path) as f:
            self.labels = [line.strip() for line in f if line.strip()]

        self.model = Darknet(str(config_path))
        self.model.load_weights(str(weights_path))
        self.model.train(False)   # set to inference mode
        self.model_width = self.model.model_width

    def detect(
        self,
        bgr_frame: np.ndarray,
        obj_thresh: float = 0.8,
        nms_thresh: float = 0.4,
    ) -> List[Detection]:
        H, W = bgr_frame.shape[:2]
        pil_img = Image.fromarray(cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB))
        H_new, W_new, scale = scale_numbers(H, W, self.model_width)
        img = pil_img.resize((W_new, H_new))
        img_np = np.array(img).astype(np.float32)
        img_np, pad_sizes = letterbox_pad(img_np)

        img_t = (
            torch.from_numpy(img_np.transpose(2, 0, 1) / 255.0)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            predictions, _ = self.model(img_t, device=self.device)

        predictions = objectness_filter_and_nms(
            predictions, self.model.classes, obj_thresh, nms_thresh
        )

        if predictions is None:
            return []

        pad_top, _, pad_left, _ = pad_sizes
        predictions[:, 0] = (predictions[:, 0] - pad_left) / scale
        predictions[:, 1] = (predictions[:, 1] - pad_top) / scale
        predictions[:, 2] = predictions[:, 2] / scale
        predictions[:, 3] = predictions[:, 3] / scale

        tl_x, tl_y, br_x, br_y = get_corner_coords(predictions)

        detections = []
        for i in range(len(predictions)):
            score, class_id = torch.max(
                predictions[i, 5: 5 + self.model.classes], dim=-1
            )
            cid = int(class_id)
            name = self.labels[cid] if cid < len(self.labels) else str(cid)
            detections.append(Detection(
                class_id=cid,
                class_name=name,
                confidence=float(score),
                x1=max(0, int(tl_x[i])),
                y1=max(0, int(tl_y[i])),
                x2=min(W, int(br_x[i])),
                y2=min(H, int(br_y[i])),
            ))

        return detections
