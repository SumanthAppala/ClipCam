from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm

from config import INDEX_NAME_FMT
from .encoder import ClipEncoder
from .common import ensure_dir, chunks
from .video_io import video_duration_seconds, timestamps_for_sampling, grab_frame_at_sec

def index_path_for(video_path: Path, out_dir: Path, model_name: str, pretrained: str) -> Path:
    return out_dir / INDEX_NAME_FMT.format(model=model_name, pretrained=pretrained)

def build_index_npz(video_path: Path, out_dir: Path, sample_fps: float,
                    model_name: str, pretrained: str) -> Path:
    ensure_dir(out_dir)
    idx_path = index_path_for(video_path, out_dir, model_name, pretrained)
    if idx_path.exists():
        print(f"[Index] Exists: {idx_path}")
        return idx_path

    duration = video_duration_seconds(video_path)
    ts = timestamps_for_sampling(duration, sample_fps)
    if not ts:
        raise RuntimeError("No timestamps generated (video may be empty).")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    encoder = ClipEncoder(model_name=model_name, pretrained=pretrained)
    all_embeds = []
    CHUNK_SECS = 64
    pbar = tqdm(total=len(ts), desc=f"Embedding @ {sample_fps} fps")
    for t_chunk in chunks(ts, CHUNK_SECS):
        images = []
        for t in t_chunk:
            im = grab_frame_at_sec(cap, t)
            images.append(im if im is not None else Image.new("RGB", (224, 224), (0, 0, 0)))
            pbar.update(1)
        embeds = encoder.encode_images(images)
        all_embeds.append(embeds)
    pbar.close()
    cap.release()

    E = np.vstack(all_embeds).astype("float32")
    ts_arr = np.array(ts, dtype=np.float32)
    np.savez_compressed(
        idx_path,
        E=E, ts=ts_arr,
        model=model_name, pretrained=pretrained,
        sample_fps=float(sample_fps)
    )
    print(f"[Index] Saved: {idx_path}  (frames: {len(ts_arr)}, dim: {E.shape[1]})")
    return idx_path

def load_index_npz(idx_path: Path):
    d = np.load(idx_path, allow_pickle=False)
    E = d["E"].astype("float32")
    ts = d["ts"].astype("float32")
    meta = {
        "model": str(d["model"]),
        "pretrained": str(d["pretrained"]),
        "sample_fps": float(d["sample_fps"]),
    }
    return E, ts, meta
