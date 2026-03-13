# Object Tracker UI Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a real-time PyQt5 desktop app that displays a live webcam feed, detects objects using the existing YOLOv3 model, and lets the user select one object for high-accuracy CSRT+YOLOv3 tracking.

**Architecture:** A `ProcessingWorker` QThread handles camera capture, YOLOv3 inference (detection mode), and CSRT tracking with async YOLOv3 re-calibration (tracking mode). The `MainWindow` renders annotated frames, manages a sidebar detection list, and routes clicks to the worker via `select_target()`. Two new files are created: `tracker_worker.py` and `tracker_app.py`. Existing `darknet.py` and `utils.py` are unchanged.

**Tech Stack:** PyQt5, opencv-python, torch (existing), PIL (existing), Python 3.x

---

## Task 1: Install dependencies and verify environment

**Files:** No new files.

**Step 1: Install PyQt5 and opencv-python**

```
pip install PyQt5 opencv-python
```

Expected: Both install without errors.

**Step 2: Verify camera and imports work**

```
python -c "import cv2; import PyQt5; cap = cv2.VideoCapture(0); print('opened:', cap.isOpened()); cap.release()"
```

Expected: `opened: True`. If 0 fails, try index 1.

**Step 3: Verify existing model loads**

```
python -c "
import torch
from darknet import Darknet
m = Darknet('./cfg/yolov3_416x416.cfg')
m.load_weights('./weights/yolov3.weights')
m.train(False)
print('Classes:', m.classes, 'Width:', m.model_width)
"
```

Expected: `Classes: 80 Width: 416`

---

## Task 2: Create `tracker_worker.py` — `Detection`, `FrameResult`, `InferenceEngine`, `find_best_matching_detection`

**Files:**
- Create: `tracker_worker.py`
- Create: `tests/test_tracker_worker.py`

**Step 1: Create `tests/` directory and write failing tests**

Run: `mkdir tests`

Create `tests/test_tracker_worker.py`:

```python
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
```

**Step 2: Run tests to verify they fail (module not found)**

```
python -m pytest tests/test_tracker_worker.py -v 2>&1 | head -20
```

Expected: `ModuleNotFoundError: No module named 'tracker_worker'`

**Step 3: Create `tracker_worker.py` (Detection, FrameResult, InferenceEngine, find_best_matching_detection)**

Create `tracker_worker.py` with the following content:

```python
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
        self.model.train(False)   # inference mode
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
```

**Step 4: Run unit tests (skip inference tests for speed)**

```
python -m pytest tests/test_tracker_worker.py -v -k "not inference_engine"
```

Expected: 5 tests PASS.

**Step 5: Run full tests including inference**

```
python -m pytest tests/test_tracker_worker.py -v
```

Expected: 7 tests PASS (inference tests may take 10-30 sec).

**Step 6: Commit**

```
git init
git add tracker_worker.py tests/test_tracker_worker.py
git commit -m "feat: add Detection, InferenceEngine, and IoU matching utility"
```

---

## Task 3: Add `ProcessingWorker` QThread to `tracker_worker.py`

**Files:**
- Modify: `tracker_worker.py` (append class)

**Step 1: Append `ProcessingWorker` to the bottom of `tracker_worker.py`**

```python
class ProcessingWorker(QThread):
    """
    State machine:
        DETECTING -> (select_target called) -> TRACKING
        TRACKING  -> (stop_tracking called)  -> DETECTING
        TRACKING  -> (target lost 30 frames) -> DETECTING
    """

    frame_ready = pyqtSignal(object)

    MODE_DETECTING = 'DETECTING'
    MODE_TRACKING = 'TRACKING'
    REDETECT_INTERVAL = 15
    LOST_THRESHOLD = 30

    def __init__(self, engine: InferenceEngine, camera_index: int = 0):
        super().__init__()
        self.engine = engine
        self.camera_index = camera_index
        self._lock = threading.Lock()
        self._running = False
        self._mode = self.MODE_DETECTING
        self.obj_thresh = 0.8

        self._tracker = None
        self._target_class: Optional[str] = None
        self._frames_since_redetect = 0
        self._frames_lost = 0
        self._latest_detections: List[Detection] = []

        self._executor = ThreadPoolExecutor(max_workers=1)
        self._pending_future = None
        self._frame_times: List[float] = []

    # --- Public API (UI thread) ---

    def select_target(self, detection_index: int, frame: np.ndarray):
        with self._lock:
            if not (0 <= detection_index < len(self._latest_detections)):
                return
            det = self._latest_detections[detection_index]
            self._tracker = cv2.TrackerCSRT_create()
            self._tracker.init(frame, det.to_csrt_rect())
            self._target_class = det.class_name
            self._frames_since_redetect = 0
            self._frames_lost = 0
            self._mode = self.MODE_TRACKING

    def stop_tracking(self):
        with self._lock:
            self._mode = self.MODE_DETECTING
            self._tracker = None
            self._target_class = None
            self._pending_future = None

    def set_obj_thresh(self, value: float):
        self.obj_thresh = max(0.1, min(0.99, value))

    def request_redetect(self):
        with self._lock:
            self._frames_since_redetect = self.REDETECT_INTERVAL

    # --- Thread entry point ---

    def run(self):
        self._running = True
        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            return

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            now = time.monotonic()
            self._frame_times.append(now)
            self._frame_times = [t for t in self._frame_times if now - t < 1.0]
            fps = float(len(self._frame_times))

            with self._lock:
                mode = self._mode

            if mode == self.MODE_DETECTING:
                result = self._run_detecting(frame, fps)
            else:
                result = self._run_tracking(frame, fps)

            self.frame_ready.emit(result)

        cap.release()

    def stop(self):
        self._running = False
        self._executor.shutdown(wait=False)
        self.wait()

    # --- Internal ---

    def _run_detecting(self, frame, fps):
        detections = self.engine.detect(frame, obj_thresh=self.obj_thresh)
        with self._lock:
            self._latest_detections = detections
        return FrameResult(
            frame=frame.copy(), detections=detections,
            tracked_rect=None, tracked_class=None,
            mode=self.MODE_DETECTING, fps=fps,
        )

    def _run_tracking(self, frame, fps):
        with self._lock:
            tracker = self._tracker

        if tracker is None:
            self.stop_tracking()
            return self._run_detecting(frame, fps)

        success, rect = tracker.update(frame)

        lost = False
        do_redetect = False

        with self._lock:
            if not success:
                self._frames_lost += 1
                lost = self._frames_lost >= self.LOST_THRESHOLD
            else:
                self._frames_lost = 0
                self._frames_since_redetect += 1
                do_redetect = (
                    self._frames_since_redetect >= self.REDETECT_INTERVAL
                )

        if lost:
            self.stop_tracking()
            return self._run_detecting(frame, fps)

        if success and do_redetect:
            if self._pending_future is None or self._pending_future.done():
                rect_snap = tuple(int(v) for v in rect)
                self._pending_future = self._executor.submit(
                    self._async_redetect, frame.copy(), rect_snap
                )
            with self._lock:
                self._frames_since_redetect = 0

        if self._pending_future and self._pending_future.done():
            new_rect = self._pending_future.result()
            self._pending_future = None
            if new_rect is not None:
                with self._lock:
                    self._tracker = cv2.TrackerCSRT_create()
                    self._tracker.init(frame, new_rect)
                rect = new_rect

        with self._lock:
            latest = list(self._latest_detections)
            target_class = self._target_class

        return FrameResult(
            frame=frame.copy(),
            detections=latest,
            tracked_rect=tuple(int(v) for v in rect) if success else None,
            tracked_class=target_class,
            mode=self.MODE_TRACKING,
            fps=fps,
        )

    def _async_redetect(self, frame, current_rect):
        detections = self.engine.detect(frame, obj_thresh=self.obj_thresh)
        with self._lock:
            self._latest_detections = detections
            target_class = self._target_class
        same_class = [d for d in detections if d.class_name == target_class]
        best = find_best_matching_detection(current_rect, same_class, min_iou=0.3)
        return best.to_csrt_rect() if best else None
```

**Step 2: Verify prior tests still pass**

```
python -m pytest tests/test_tracker_worker.py -v -k "not inference_engine"
```

Expected: 5 tests PASS.

**Step 3: Smoke-test the worker without UI**

```
python -c "
import sys, time
from PyQt5.QtCore import QCoreApplication
from tracker_worker import InferenceEngine, ProcessingWorker
app = QCoreApplication(sys.argv)
engine = InferenceEngine('./weights/yolov3.weights', './cfg/yolov3_416x416.cfg', './data/coco.names')
worker = ProcessingWorker(engine, camera_index=0)
results = []
worker.frame_ready.connect(lambda r: results.append(r))
worker.start()
time.sleep(4)
worker.stop()
print(f'Frames: {len(results)}, Last mode: {results[-1].mode if results else None}')
"
```

Expected: `Frames: N, Last mode: DETECTING`

**Step 4: Commit**

```
git add tracker_worker.py
git commit -m "feat: add ProcessingWorker QThread with CSRT and async YOLOv3 re-detection"
```

---

## Task 4: Create `tracker_app.py` — full PyQt5 application

**Files:**
- Create: `tracker_app.py`

**Step 1: Create `tracker_app.py`**

```python
import sys
from pathlib import Path

import cv2
import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QPushButton, QSizePolicy,
    QSlider, QStatusBar, QVBoxLayout, QWidget,
)

from tracker_worker import FrameResult, InferenceEngine, ProcessingWorker

WEIGHTS_PATH = Path('./weights/yolov3.weights')
CONFIG_PATH  = Path('./cfg/yolov3_416x416.cfg')
LABELS_PATH  = Path('./data/coco.names')

DARK_STYLE = """
    QMainWindow, QWidget { background: #121212; color: #dddddd; }
    QLabel { color: #dddddd; }
    QListWidget {
        background: #1e1e1e; color: #dddddd; border: 1px solid #333333;
    }
    QListWidget::item:selected { background: #0078d4; }
    QListWidget::item:hover    { background: #2a2a2a; }
    QPushButton { border-radius: 4px; padding: 6px 10px; font-size: 12px; }
    QSlider::groove:horizontal {
        height: 4px; background: #333333; border-radius: 2px;
    }
    QSlider::handle:horizontal {
        background: #0078d4; width: 14px; height: 14px;
        margin: -5px 0; border-radius: 7px;
    }
    QStatusBar { background: #1a1a1a; color: #888888; }
"""


class VideoLabel(QLabel):
    """QLabel that maps mouse clicks to video-frame coordinates."""

    object_clicked = pyqtSignal(int, int)   # frame_x, frame_y

    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(640, 480)
        self.setCursor(Qt.CrossCursor)
        self._fw, self._fh = 640, 480

    def update_frame_size(self, w, h):
        self._fw, self._fh = w, h

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return
        scale = min(self.width() / self._fw, self.height() / self._fh)
        dw = int(self._fw * scale)
        dh = int(self._fh * scale)
        ox = (self.width()  - dw) // 2
        oy = (self.height() - dh) // 2
        cx = event.x() - ox
        cy = event.y() - oy
        if 0 <= cx < dw and 0 <= cy < dh:
            self.object_clicked.emit(int(cx / scale), int(cy / scale))


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Object Tracker")
        self.resize(1100, 700)
        self._current_frame = None
        self._current_detections = []
        self._build_ui()
        self._start_worker()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Video
        self.video_label = VideoLabel()
        self.video_label.object_clicked.connect(self._on_video_clicked)
        root.addWidget(self.video_label, stretch=3)

        # Sidebar
        sidebar = QWidget()
        sidebar.setFixedWidth(230)
        col = QVBoxLayout(sidebar)
        col.setContentsMargins(0, 0, 0, 0)
        col.setSpacing(6)

        col.addWidget(_header("DETECTED OBJECTS"))
        self.detection_list = QListWidget()
        self.detection_list.itemClicked.connect(self._on_list_item_clicked)
        col.addWidget(self.detection_list, stretch=2)

        col.addWidget(_divider())
        col.addWidget(_header("TRACKING"))

        self.target_label = QLabel("Target: —")
        self.target_label.setStyleSheet("color: #00aaff; font-size: 12px;")
        col.addWidget(self.target_label)

        self.position_label = QLabel("Position: —")
        self.position_label.setStyleSheet("color: #888888; font-size: 11px;")
        col.addWidget(self.position_label)

        self.stop_btn = QPushButton("Stop Tracking")
        self.stop_btn.setEnabled(False)
        self.stop_btn.setStyleSheet(
            "QPushButton { background: #c0392b; color: white; }"
        )
        self.stop_btn.clicked.connect(self._on_stop_tracking)
        col.addWidget(self.stop_btn)

        self.redetect_btn = QPushButton("Re-detect Now")
        self.redetect_btn.setStyleSheet(
            "QPushButton { background: #27ae60; color: white; }"
        )
        self.redetect_btn.clicked.connect(self._on_redetect)
        col.addWidget(self.redetect_btn)

        col.addWidget(_divider())
        col.addWidget(_header("CONFIDENCE THRESHOLD"))

        self.thresh_val_lbl = QLabel("0.80")
        self.thresh_val_lbl.setStyleSheet("color: #dddddd; font-size: 11px;")
        col.addWidget(self.thresh_val_lbl)

        self.thresh_slider = QSlider(Qt.Horizontal)
        self.thresh_slider.setRange(10, 99)
        self.thresh_slider.setValue(80)
        self.thresh_slider.valueChanged.connect(self._on_thresh_changed)
        col.addWidget(self.thresh_slider)

        col.addStretch()
        root.addWidget(sidebar)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Mode: DETECTING | FPS: 0 | Camera: 0")

    # ------------------------------------------------------------------
    # Worker
    # ------------------------------------------------------------------

    def _start_worker(self):
        engine = InferenceEngine(WEIGHTS_PATH, CONFIG_PATH, LABELS_PATH)
        self.worker = ProcessingWorker(engine, camera_index=0)
        self.worker.frame_ready.connect(self._on_frame_ready)
        self.worker.start()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_frame_ready(self, result: FrameResult):
        self._current_frame = result.frame.copy()
        self._current_detections = result.detections

        rendered = _render_frame(result)
        _show_frame(self.video_label, rendered)

        if result.mode == 'DETECTING':
            self._refresh_list(result.detections)
            self.target_label.setText("Target: —")
            self.position_label.setText("Position: —")
            self.stop_btn.setEnabled(False)
        else:
            self.stop_btn.setEnabled(True)
            self.target_label.setText(f"Target: {result.tracked_class or '—'}")
            if result.tracked_rect:
                x, y, w, h = result.tracked_rect
                self.position_label.setText(
                    f"Position: {x + w//2}, {y + h//2}\nSize: {w}×{h}"
                )

        self.status_bar.showMessage(
            f"Mode: {result.mode} | FPS: {int(result.fps)} | Camera: 0"
        )

    def _on_video_clicked(self, fx, fy):
        for i, det in enumerate(self._current_detections):
            if det.x1 <= fx <= det.x2 and det.y1 <= fy <= det.y2:
                self._start_tracking(i)
                return

    def _on_list_item_clicked(self, item: QListWidgetItem):
        self._start_tracking(item.data(Qt.UserRole))

    def _start_tracking(self, idx: int):
        if self._current_frame is not None:
            self.worker.select_target(idx, self._current_frame)

    def _on_stop_tracking(self):
        self.worker.stop_tracking()
        self.stop_btn.setEnabled(False)

    def _on_redetect(self):
        self.worker.request_redetect()

    def _on_thresh_changed(self, value: int):
        thresh = value / 100.0
        self.thresh_val_lbl.setText(f"{thresh:.2f}")
        self.worker.set_obj_thresh(thresh)

    def _refresh_list(self, detections):
        self.detection_list.clear()
        for i, det in enumerate(detections):
            item = QListWidgetItem(
                f"{det.class_name}  {det.confidence * 100:.0f}%"
            )
            item.setData(Qt.UserRole, i)
            self.detection_list.addItem(item)

    def closeEvent(self, event):
        self.worker.stop()
        super().closeEvent(event)


# ------------------------------------------------------------------
# Pure rendering helpers
# ------------------------------------------------------------------

def _render_frame(result: FrameResult) -> np.ndarray:
    frame = result.frame.copy()
    is_tracking = result.mode == 'TRACKING'

    for det in result.detections:
        color = (0, 100, 0) if is_tracking else (0, 220, 0)
        thick = 1 if is_tracking else 2
        cv2.rectangle(frame, (det.x1, det.y1), (det.x2, det.y2), color, thick)
        cv2.putText(
            frame,
            f"{det.class_name} {det.confidence * 100:.0f}%",
            (det.x1, max(det.y1 - 4, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
        )

    if is_tracking and result.tracked_rect:
        x, y, w, h = result.tracked_rect
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)
        cv2.putText(
            frame,
            f"TRACKING: {result.tracked_class}",
            (x, max(y - 8, 16)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2,
        )
        cv2.drawMarker(
            frame, (x + w // 2, y + h // 2),
            (255, 255, 0), cv2.MARKER_CROSS, 20, 2,
        )

    return frame


def _show_frame(label: VideoLabel, bgr_frame: np.ndarray):
    h, w = bgr_frame.shape[:2]
    label.update_frame_size(w, h)
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    qimg = QImage(rgb.data, w, h, w * 3, QImage.Format_RGB888)
    label.setPixmap(
        QPixmap.fromImage(qimg).scaled(
            label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
    )


def _header(text: str) -> QLabel:
    lbl = QLabel(text)
    lbl.setStyleSheet("font-weight: bold; color: #aaaaaa; font-size: 11px;")
    return lbl


def _divider() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet("color: #333333;")
    return line


if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setStyleSheet(DARK_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

**Step 2: Launch the app**

```
python tracker_app.py
```

Expected: Dark-themed window opens. Live camera feed visible. YOLOv3 runs and green bounding boxes appear around detected objects after a brief model-load delay (~5-10 sec on CPU).

**Step 3: Test detection → tracking via video click**

1. Wait for at least one green bounding box.
2. Click directly on the box in the video.
3. Expected: Box turns cyan with "TRACKING: classname" label. Sidebar shows target info.

**Step 4: Test detection → tracking via sidebar list**

1. Click "Stop Tracking".
2. Click any item in the "DETECTED OBJECTS" list.
3. Expected: Tracking starts on that object.

**Step 5: Test controls**

- Drag confidence threshold slider. Expected: fewer/more objects detected.
- Click "Re-detect Now". Expected: detection list refreshes.
- Close the window. Expected: app closes cleanly without hanging.

**Step 6: Commit**

```
git add tracker_app.py
git commit -m "feat: add PyQt5 tracker app with live detection, click-to-track, and sidebar"
```

---

## Task 5: Write memory and final commit

**Files:**
- Create: `C:\Users\olive\.claude\projects\C--Users-olive-Downloads-Object-Tracker1\memory\MEMORY.md`

**Step 1: Run all tests**

```
python -m pytest tests/ -v
```

Expected: All tests PASS.

**Step 2: Create memory file**

Create `C:\Users\olive\.claude\projects\C--Users-olive-Downloads-Object-Tracker1\memory\MEMORY.md`:

```markdown
# Object Tracker Project Memory

## Goal
Real-time AI object tracker → physical drone-tracking system.

## Key Files
- `tracker_app.py` — PyQt5 MainWindow, entry point (`python tracker_app.py`)
- `tracker_worker.py` — ProcessingWorker, InferenceEngine, Detection, FrameResult
- `darknet.py` — YOLOv3 model (DO NOT MODIFY)
- `utils.py` — inference utilities (DO NOT MODIFY)
- `cfg/yolov3_416x416.cfg` — active config (faster than 608 version)
- `weights/yolov3.weights` — COCO pretrained, 80 classes
- `data/coco.names` — class labels

## Architecture
- ProcessingWorker (QThread): camera + CSRT tracking + async YOLOv3 re-detection
- State: DETECTING <-> TRACKING
- CSRT every frame; YOLOv3 re-detect every 15 frames async (ThreadPoolExecutor)
- Lost threshold: 30 CSRT failures -> auto-return to DETECTING

## Tests
- `tests/test_tracker_worker.py` — Detection, IoU matching, InferenceEngine smoke tests

## Dependencies Added
- PyQt5
- opencv-python
```

**Step 3: Final commit**

```
git add docs/ tests/ memory/
git commit -m "docs: add design docs, implementation plan, and project memory"
```
