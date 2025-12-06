'''import argparse
from pathlib import Path
import numpy as np

from config import (
    MODEL_NAME, PRETRAINED, SAMPLE_FPS, WINDOWS_DEFAULT, MIN_GAP_SEC,
    DATA_ROOT, RESULTS_ROOT, THUMBS_DIRNAME, CLIPS_DIRNAME, RESULTS_CSV_FMT
)
from utils.common import ensure_dir, slugify
from utils.encoder import ClipEncoder
from utils.indexer import build_index_npz, load_index_npz, index_path_for
from utils.search import per_frame_scores, pick_top_segments
from utils.export import save_thumbnail, export_clips, save_keyframes

def _seconds_from_index(ts: np.ndarray, idx: int) -> float:
    idx = max(0, min(idx, len(ts)-1))
    return float(ts[idx])

def cmd_index(args):
    video = Path(args.video)
    if not video.exists():
        # allow shorthand like --video my.mp4 to search under data/
        alt = DATA_ROOT / args.video
        if alt.exists():
            video = alt
        else:
            raise SystemExit(f"Video not found: {args.video}")

    vid_dir = RESULTS_ROOT / video.stem
    ensure_dir(vid_dir)
    build_index_npz(video, vid_dir, args.sample_fps, args.model, args.pretrained)

def cmd_search(args):
    video = Path(args.video)
    if not video.exists():
        alt = DATA_ROOT / args.video
        if alt.exists():
            video = alt
        else:
            raise SystemExit(f"Video not found: {args.video}")

    vid_dir = RESULTS_ROOT / video.stem
    ensure_dir(vid_dir)

    # ensure index exists
    idx_path = index_path_for(video, vid_dir, args.model, args.pretrained)
    if not idx_path.exists():
        build_index_npz(video, vid_dir, args.sample_fps, args.model, args.pretrained)

    E, ts, meta = load_index_npz(idx_path)
    sample_fps = float(meta["sample_fps"])

    encoder = ClipEncoder(model_name=args.model, pretrained=args.pretrained)
    tvec = encoder.encode_text(args.query)
    S = per_frame_scores(E, tvec)

    # seconds -> frames
    windows_frames = [max(1, int(round(w * sample_fps))) for w in args.windows]
    min_gap_frames = max(0, int(round(args.min_gap * sample_fps)))

    segs_idx = pick_top_segments(S, windows_frames=windows_frames,
                                 topk=args.topk, min_gap_frames=min_gap_frames)

    # Render outputs
    thumbs_dir = vid_dir / THUMBS_DIRNAME
    clips_dir  = vid_dir / CLIPS_DIRNAME

    results = []
    for (a, b, sc, Wf) in segs_idx:
        t0 = _seconds_from_index(ts, a)
        t1 = _seconds_from_index(ts, b - 1)
        W_sec = int(round(Wf / sample_fps))
        results.append((t0, t1, float(sc), W_sec))
        save_thumbnail(video, t0, thumbs_dir)

    # CSV
    slug = slugify(args.query)[:80]
    csv_path = vid_dir / RESULTS_CSV_FMT.format(slug=slug)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("start_sec,end_sec,score,window_s\n")
        for t0, t1, sc, W in results:
            f.write(f"{t0:.2f},{t1:.2f},{sc:.6f},{W}\n")

    print(f"[Results] CSV: {csv_path}")
    print(f"[Results] Thumbs: {thumbs_dir}")
    for i, (t0, t1, sc, W) in enumerate(results, 1):
        print(f"{i:2d}. {t0:7.2f}s → {t1:7.2f}s  (W={W:>2}s)  score={sc:.4f}")

    # optional MP4 exports
    if args.save_clips:
        export_clips(video, results, clips_dir, method=args.clip_method, reencode=args.reencode)
    # Also save first/last frame if requested
    if args.save_keyframes:
        save_keyframes(video, thumbs_dir)

def main():
    ap = argparse.ArgumentParser(prog="clipmoments", description="Text-to-video moment search (CLIP MVP)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_idx = sub.add_parser("index", help="Build CLIP frame index for a video")
    ap_idx.add_argument("--video", required=True, type=str, help="Path or filename (searched in data/)")
    ap_idx.add_argument("--sample-fps", type=float, default=SAMPLE_FPS)
    ap_idx.add_argument("--model", type=str, default=MODEL_NAME)
    ap_idx.add_argument("--pretrained", type=str, default=PRETRAINED)
    ap_idx.set_defaults(func=cmd_index)

    ap_s = sub.add_parser("search", help="Search a video for a natural-language moment")
    ap_s.add_argument("--video", required=True, type=str, help="Path or filename (searched in data/)")
    ap_s.add_argument("--query", required=True, type=str)
    ap_s.add_argument("--windows", type=float, nargs="*", default=WINDOWS_DEFAULT, help="Window sizes (seconds)")
    ap_s.add_argument("--min-gap", type=float, default=MIN_GAP_SEC, help="Min separation between picks (sec)")
    ap_s.add_argument("--topk", type=int, default=5)
    ap_s.add_argument("--sample-fps", type=float, default=SAMPLE_FPS, help="Used if index not found")
    ap_s.add_argument("--model", type=str, default=MODEL_NAME)
    ap_s.add_argument("--pretrained", type=str, default=PRETRAINED)
    ap_s.add_argument("--save-clips", action="store_true")
    ap_s.add_argument("--clip-method", choices=["auto","ffmpeg","opencv"], default="auto")
    ap_s.add_argument("--reencode", action="store_true")
    ap_s.add_argument("--save-keyframes", action="store_true",
                  help="Also save first_frame.jpg and last_frame.jpg in results/<video>/thumbs")
    ap_s.set_defaults(func=cmd_search)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
'''
import streamlit as st
from PIL import Image
from streamlit_lottie import st_lottie
import json
import pages.clipcam_search as cs

# Page setup
st.set_page_config(page_title="The Case of ClipCam", layout="wide")

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "home"

# --- Navigation ---
if st.session_state.page == "search":
    cs.main()  # show only the search page
else:
    # --- Home page content ---
    logo = Image.open("assets/image_logo.jpg")
    with open("assets/detective search.json", "r") as f:
        detective_walk = json.load(f)

    st.markdown("""
    <h1 style="text-align:center; font-size:60px; margin-top:20px;">
    🕵️ The Case of ClipCam
    </h1>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.25, 2, 1])
    with col2:
        st.image("assets/image_logo-modified.png", width=400)

    st_lottie(detective_walk, height=350, key="detective_walk")

    st.markdown("""
    <p style="text-align:center; font-size:22px; color: #777; margin-top:20px;">
    Where detectives uncover hidden moments inside videos
    </p>
    """, unsafe_allow_html=True)

    # Button slightly to the right
    col1, col2, col3 = st.columns([2, 2, 1])
    with col2:
        clicked = st.button("🔎 Begin the Investigation")
        if clicked:
            st.session_state.page = "search"
