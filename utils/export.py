from pathlib import Path
from typing import List, Tuple
import subprocess, shutil
import cv2
from PIL import Image

from config import THUMBS_DIRNAME, CLIPS_DIRNAME
from .common import ensure_dir
from .video_io import grab_frame_at_sec
from .video_io import grab_frame_at_sec, video_duration_seconds

def save_thumbnail(video_path: Path, t_sec: float, thumbs_dir: Path, size=(480, 270)) -> Path:
    ensure_dir(thumbs_dir)
    out_path = thumbs_dir / f"thumb_{int(t_sec)}s.jpg"
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return out_path
    im = grab_frame_at_sec(cap, t_sec)
    cap.release()
    if im is None:
        return out_path
    im = im.copy()
    im.thumbnail(size)
    im.save(out_path, quality=90)
    return out_path

def save_keyframes(video_path: Path, thumbs_dir: Path, size=(480, 270)):
    """
    Saves first_frame.jpg and last_frame.jpg into thumbs_dir.
    Uses a small offset from the exact end to avoid EOF issues.
    """
    ensure_dir(thumbs_dir)

    first_path = thumbs_dir / "first_frame.jpg"
    last_path  = thumbs_dir / "last_frame.jpg"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Keyframes] Could not open video: {video_path}")
        return first_path, last_path

    # First frame
    im0 = grab_frame_at_sec(cap, 0.0)
    if im0 is not None:
        im = im0.copy()
        if size:
            im.thumbnail(size)
        im.save(first_path, quality=90)

    # Last frame (slightly before duration to avoid boundary read)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = video_duration_seconds(video_path)
    t_last = max(0.0, duration - (1.0 / fps))
    im1 = grab_frame_at_sec(cap, t_last)
    if im1 is None:
        # one more try a bit earlier
        im1 = grab_frame_at_sec(cap, max(0.0, t_last - 0.5))
    if im1 is not None:
        im = im1.copy()
        if size:
            im.thumbnail(size)
        im.save(last_path, quality=90)

    cap.release()
    print(f"[Keyframes] Saved: {first_path.name}, {last_path.name} -> {thumbs_dir}")
    return first_path, last_path

def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None

def save_clip_ffmpeg(input_path: Path, start: float, end: float, out_path: Path, reencode=False):
    ensure_dir(out_path.parent)
    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(input_path)]
    if reencode:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", "-movflags", "+faststart"]
    else:
        cmd += ["-c", "copy"]
    cmd += [str(out_path)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def save_clip_opencv(input_path: Path, start: float, end: float, out_path: Path):
    ensure_dir(out_path.parent)
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    h  = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 360)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
    start_f = int(start * fps); end_f = int(end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_f))
    f = start_f
    while f < end_f:
        ok, frame = cap.read()
        if not ok or frame is None: break
        out.write(frame)
        f += 1
    out.release(); cap.release()

def export_clips(video_path: Path, segments: List[Tuple[float,float,float,int]],
                 out_dir: Path, method="auto", reencode=False) -> List[Path]:
    """
    segments: list of (start_sec, end_sec, score, W_sec)
    Saves MP4s into out_dir
    """
    ensure_dir(out_dir)
    use_ffmpeg = (method == "ffmpeg") or (method == "auto" and have_ffmpeg())
    paths = []
    for i, (t0, t1, sc, W) in enumerate(segments, 1):
        clip_path = out_dir / f"clip_{i:02d}_{int(t0)}s_{int(t1)}s.mp4"
        try:
            if use_ffmpeg:
                save_clip_ffmpeg(video_path, t0, t1, clip_path, reencode=reencode)
            else:
                save_clip_opencv(video_path, t0, t1, clip_path)
            paths.append(clip_path)
        except Exception as e:
            print(f"[Clip] Failed to save {clip_path.name}: {e}")
    print(f"[Clips] Saved {len(paths)} clip(s) to: {out_dir}")
    return paths
