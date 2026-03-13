import sys
import numpy as np
import pytest

sys.path.insert(0, '.')
from tracker_worker import Detection, find_best_matching_detection, InferenceEngine


# --- Detection dataclass ---

def test_detection_properties():
    det = Detection(class_id=0, class_name='person', confidence=0.95,
                    x1=10, y1=20, x2=110, y2=220)
    assert det.cx == 60
    assert det.cy == 120
    assert det.w == 100
    assert det.h == 200
    assert det.to_csrt_rect() == (10, 20, 100, 200)


# --- find_best_matching_detection ---

def _det(x1, y1, x2, y2, name='person'):
    return Detection(0, name, 0.9, x1, y1, x2, y2)


def test_find_best_match_perfect_overlap():
    rect = (10, 10, 100, 100)
    detections = [_det(10, 10, 110, 110)]
    result = find_best_matching_detection(rect, detections, min_iou=0.3)
    assert result is not None
    assert result.x1 == 10


def test_find_best_match_no_overlap():
    rect = (0, 0, 50, 50)
    detections = [_det(200, 200, 300, 300)]
    result = find_best_matching_detection(rect, detections, min_iou=0.3)
    assert result is None


def test_find_best_match_empty_detections():
    result = find_best_matching_detection((10, 10, 100, 100), [], min_iou=0.3)
    assert result is None


def test_find_best_match_selects_highest_iou():
    rect = (10, 10, 100, 100)
    det1 = _det(60, 60, 160, 160)   # small overlap
    det2 = _det(15, 15, 105, 105)   # large overlap
    result = find_best_matching_detection(rect, [det1, det2], min_iou=0.3)
    assert result is det2


def test_find_best_match_below_min_iou_returns_none():
    rect = (0, 0, 10, 10)
    detections = [_det(8, 8, 100, 100)]   # tiny overlap
    result = find_best_matching_detection(rect, detections, min_iou=0.5)
    assert result is None


# --- InferenceEngine smoke test (requires model weights) ---

def test_inference_engine_returns_list():
    try:
        engine = InferenceEngine(
            weights_path='./weights/yolov3.weights',
            config_path='./cfg/yolov3_416x416.cfg',
            labels_path='./data/coco.names',
        )
    except Exception as e:
        pytest.skip(f"Model not available: {e}")
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result = engine.detect(frame, obj_thresh=0.8)
    assert isinstance(result, list)


def test_inference_engine_detects_dog():
    import cv2
    try:
        engine = InferenceEngine(
            weights_path='./weights/yolov3.weights',
            config_path='./cfg/yolov3_416x416.cfg',
            labels_path='./data/coco.names',
        )
    except Exception as e:
        pytest.skip(f"Model not available: {e}")
    frame = cv2.imread('./data/dog.jpg')
    if frame is None:
        pytest.skip("Sample image not found")
    detections = engine.detect(frame, obj_thresh=0.5)
    names = [d.class_name for d in detections]
    assert any('dog' in n for n in names), f"Expected 'dog' in {names}"
