# Object Tracker UI Design
**Date:** 2026-03-13

## Goal
Build a real-time object detection and tracking desktop application using the existing YOLOv3 implementation. The user sees a live webcam feed, detected objects are listed and highlighted, and the user can select one object (by clicking the video or the sidebar list) for high-accuracy continuous tracking.

---

## Architecture

Three threads:

1. **Capture thread** — OpenCV reads webcam frames and pushes them to a thread-safe queue.
2. **Inference/tracking thread** — Consumes frames. Runs YOLOv3 in detection mode or CSRT+YOLOv3 in tracking mode. Emits processed frame + metadata via Qt signal.
3. **PyQt5 UI thread** — Renders frames, manages sidebar list, handles user input.

### State machine
```
DETECTING  -->  user selects object  -->  TRACKING
TRACKING   -->  user clicks Stop     -->  DETECTING
TRACKING   -->  target lost (30f)    -->  DETECTING
```

---

## UI Layout

```
+--------------------------------------------------+----------------------+
|                                                  |  DETECTED OBJECTS    |
|              LIVE CAMERA FEED                    | > person  95%        |
|                                                  |   car     87%        |
|   [green bbox: detected objects]                 |   bird    72%        |
|   [cyan bbox + "TRACKING": selected object]      |                      |
|                                                  |  TRACKING            |
|                                                  |  Target: person      |
|                                                  |  Confidence: 95%     |
|                                                  |  Position: 320, 240  |
|                                                  | [  Stop Tracking  ]  |
|                                                  | [ Re-detect Now   ]  |
+--------------------------------------------------+----------------------+
|  Mode: TRACKING | FPS: 28 | Camera: 0            |                      |
+--------------------------------------------------+----------------------+
```

- Click bounding box on video OR click row in sidebar list → start tracking
- "Stop Tracking" → return to detection mode
- "Re-detect Now" → force immediate YOLOv3 pass
- Confidence threshold slider (default 0.8)

---

## Tracking Accuracy: Two-Layer Approach

### Layer 1 — CSRT tracker (every frame)
- OpenCV `cv2.TrackerCSRT_create()` — highest accuracy single-object tracker in OpenCV
- Handles rotation, scale changes, partial occlusion
- No neural network cost per frame → runs at full camera FPS

### Layer 2 — YOLOv3 re-detection (every 15 frames)
- Re-runs YOLOv3 and IoU-matches result to current tracker position
- Good match → re-initializes CSRT with fresh bbox (corrects drift)
- No match for 30 consecutive frames → target lost → return to DETECTING

---

## Files

| File | Purpose |
|------|---------|
| `tracker_app.py` | New entry point — PyQt5 app |
| `tracker_worker.py` | Capture + inference/tracking thread logic |
| `darknet.py` | Existing — unchanged |
| `utils.py` | Existing — unchanged |

---

## Dependencies (new)
- `opencv-python` — camera capture, CSRT tracker
- `PyQt5` — UI framework
