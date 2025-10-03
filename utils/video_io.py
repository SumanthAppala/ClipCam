from pathlib import Path
from typing import Tuple, List, Optional
import cv2
from PIL import Image

def video_meta(path: Path) -> Tuple[float, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_cnt = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return float(fps), frame_cnt

def video_duration_seconds(path: Path) -> float:
    fps, n = video_meta(path)
    return 0.0 if fps <= 0 or n <= 0 else n / fps

def timestamps_for_sampling(duration_sec: float, sample_fps: float) -> List[float]:
    if duration_sec <= 0 or sample_fps <= 0:
        return []
    step = 1.0 / sample_fps
    t, ts = 0.0, []
    while t <= duration_sec:
        ts.append(round(t, 3))
        t += step
    return ts

def grab_frame_at_sec(cap: cv2.VideoCapture, t_sec: float) -> Optional[Image.Image]:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_sec * 1000.0))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # BGR->RGB
    return Image.fromarray(frame)
