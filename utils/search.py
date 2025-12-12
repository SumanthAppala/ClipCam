from typing import Iterable, List, Tuple
import numpy as np

def per_frame_scores(E: np.ndarray, t_vec: np.ndarray) -> np.ndarray:
                                                 
    return E @ t_vec

def window_scores(prefix: np.ndarray, W: int) -> np.ndarray:
    T = len(prefix) - 1
    if W < 1 or W > T:
        return np.array([], dtype=np.float32)
    return (prefix[W:] - prefix[:-W]) / float(W)

def non_max_suppress_1d(segments: List[Tuple[int,int,float]], iou_thr=0.0) -> List[Tuple[int,int,float]]:
    if not segments: return []
    segs = sorted(segments, key=lambda x: x[2], reverse=True)
    kept = []
    def iou(a,b):
        inter = max(0, min(a[1], b[1]) - max(a[0], b[0]))
        uni = (a[1]-a[0]) + (b[1]-b[0]) - inter
        return 0.0 if uni <= 0 else inter/uni
    for s in segs:
        if all(iou(s,k) <= iou_thr for k in kept):
            kept.append(s)
    return kept

def pick_top_segments(S: np.ndarray, windows_frames: Iterable[int],
                      topk: int, min_gap_frames: int) -> List[Tuple[int,int,float,int]]:
    T = len(S)
    prefix = np.zeros(T + 1, dtype=np.float32); np.cumsum(S, out=prefix[1:])
    candidates = []
    for W in windows_frames:
        if W < 1 or W > T: continue
        w_scores = window_scores(prefix, W)
        order = np.argsort(-w_scores)
        picks = []
        for j in order:
            start, end, sc = int(j), int(j+W), float(w_scores[j])
            if all(abs(start - p[0]) >= (W + min_gap_frames) for p in picks):
                picks.append((start, end, sc, W))
            if len(picks) >= topk: break
        candidates.extend(picks)
    merged = non_max_suppress_1d([(a,b,sc) for (a,b,sc,_) in candidates], iou_thr=0.0)
    merged = sorted(merged, key=lambda x: x[2], reverse=True)[:topk]
    out = []
    for (a,b,sc) in merged:
        Ws = [W for (aa,bb,scc,W) in candidates if aa==a and bb==b and abs(scc-sc)<1e-6]
        W = Ws[0] if Ws else (b-a)
        out.append((a,b,sc,W))
    return out
