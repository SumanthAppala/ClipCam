import sys
import io
import time
import os
import json
import torch
import numpy as np
import streamlit as st
from pathlib import Path
from typing import Dict, Any, List
from PIL import Image
import google.generativeai as genai

try:
    from streamlit_lottie import st_lottie
except ImportError:
    st.error("Please install streamlit-lottie: `pip install streamlit-lottie`")
    st.stop()

                               
from config import (
    MODEL_NAME, PRETRAINED, SAMPLE_FPS, WINDOWS_DEFAULT, MIN_GAP_SEC,
    DATA_ROOT, RESULTS_ROOT, THUMBS_DIRNAME, CLIPS_DIRNAME, RESULTS_CSV_FMT
)
from utils.common import ensure_dir, slugify
from utils.encoder import ClipEncoder
from utils.indexer import build_index_npz, load_index_npz, index_path_for
from utils.search import per_frame_scores, pick_top_segments
from utils.export import save_thumbnail, export_clips, save_keyframes
from utils.detector import ObjectDetector
from utils.llm import GeminiChat
                                                                         
st.set_page_config(page_title="The Case of ClipCam", layout="wide")

api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    if "GOOGLE_API_KEY" in st.secrets:
        api_key = st.secrets["GOOGLE_API_KEY"]
    else:
        st.error("🚨 API Key not found! Please export GOOGLE_API_KEY in your terminal.")
        st.stop()

genai.configure(api_key=api_key)

@st.cache_resource
def load_detector():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return ObjectDetector(device=device)

def _seconds_from_index(ts: np.ndarray, idx: int) -> float:
    idx = max(0, min(idx, len(ts) - 1))
    return float(ts[idx])

def ensure_video_on_disk(uploaded_file) -> Path:
    ensure_dir(DATA_ROOT)
    dest = DATA_ROOT / uploaded_file.name
    with open(dest, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest

def run_index(video_path: Path, sample_fps: float, model_name: str, pretrained: str) -> Path:
    vid_dir = RESULTS_ROOT / video_path.stem
    ensure_dir(vid_dir)
    idx_path = index_path_for(video_path, vid_dir, model_name, pretrained)
    if not idx_path.exists():
        build_index_npz(video_path, vid_dir, sample_fps, model_name, pretrained)
    return idx_path

def inject_custom_css():
    st.markdown(
        """
        <style>
        /* 1. Main Background: The Detective Gradient */
        .stApp {
            background: linear-gradient(180deg, #43282b 0%, #614141 30%, #97552f 60%, #d6a055 85%, #e9cda2 100%);
            background-attachment: fixed;
            background-size: cover;
        }
        
        /* 2. Sidebar Background: Semi-transparent dark brown to match the theme */
        section[data-testid="stSidebar"] {
            background-color: rgba(67, 40, 43, 0.95);
        }

        /* 3. Text Visibility: Force headers and labels to be white with a shadow */
        h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {
            color: #ffffff !important;
            text-shadow: 2px 2px 4px #000000;
        }
        
        /* 4. Fix specific input labels being hard to read */
        .stTextInput > label, .stNumberInput > label, .stFileUploader > label {
            color: #e9cda2 !important; /* Use the lightest palette color for input labels */
            font-weight: bold;
        }
        
        /* 5. Custom Button Styling (Optional: Makes buttons look "wooden") */
        .stButton > button {
            background-color: #97552f;
            color: white;
            border: 1px solid #43282b;
        }
        .stButton > button:hover {
            background-color: #d6a055;
            color: #43282b;
            border-color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
def run_search_top1(
    video_path: Path,
    query: str,
    windows_s: List[float],
    min_gap_s: float,
    sample_fps: float,
    model_name: str,
    pretrained: str,
    reencode: bool,
    save_keyframes_flag: bool,
    draw_boxes: bool = False,
    detector = None,
) -> Dict[str, Any]:
    vid_dir = RESULTS_ROOT / video_path.stem
    ensure_dir(vid_dir)

    idx_path = run_index(video_path, sample_fps, model_name, pretrained)

    E, ts, meta = load_index_npz(idx_path)
    sample_fps = float(meta["sample_fps"])

    encoder = ClipEncoder(model_name=model_name, pretrained=pretrained)
    tvec = encoder.encode_text(query)
    S = per_frame_scores(E, tvec)

    windows_frames = [max(1, int(round(w * sample_fps))) for w in windows_s]
    min_gap_frames = max(0, int(round(min_gap_s * sample_fps)))

    segs_idx = pick_top_segments(S, windows_frames=windows_frames, topk=1, min_gap_frames=min_gap_frames)

    thumbs_dir = vid_dir / THUMBS_DIRNAME
    clips_dir = vid_dir / CLIPS_DIRNAME
    ensure_dir(thumbs_dir)
    ensure_dir(clips_dir)

    results = []
    for (a, b, sc, Wf) in segs_idx:
        t0 = _seconds_from_index(ts, a)
        t1 = _seconds_from_index(ts, b - 1)
        W_sec = int(round(Wf / sample_fps))
        results.append((t0, t1, float(sc), W_sec))
        save_thumbnail(video_path, t0, thumbs_dir)

    slug = slugify(query)[:80]
    csv_path = vid_dir / RESULTS_CSV_FMT.format(slug=slug)
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("start_sec,end_sec,score,window_s\n")
        for t0, t1, sc, W in results:
            f.write(f"{t0:.2f},{t1:.2f},{sc:.6f},{W}\n")
    
    pass_detector = detector if draw_boxes else None
    export_clips(video_path, results, clips_dir, method="opencv", reencode=reencode, detector=pass_detector, text_query=query)

    if save_keyframes_flag:
        save_keyframes(video_path, thumbs_dir)

    time.sleep(0.2)
    new_files = sorted(clips_dir.glob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
    clip_path = new_files[0] if new_files else None

    return {
        "results": results,
        "csv_path": csv_path,
        "thumbs_dir": thumbs_dir,
        "clips_dir": clips_dir,
        "clip_path": clip_path,
        "vid_dir": vid_dir,
        "idx_path": idx_path,
    }

                                                       
def render_home_page():
                                       
    try:
        with open("assets/detective search.json", "r") as f:
            detective_walk = json.load(f)
    except FileNotFoundError:
        st.warning("Animation file 'assets/detective search.json' not found.")
        detective_walk = None

           
    st.markdown("""
    <h1 style="text-align:center; font-size:60px; margin-top:20px; color: #000000;">
    🕵️ Sherlock.AI
    </h1>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.65, 1.5, 1.5 ])
    with col2:
                              
        if os.path.exists("assets/image_logo-modified.png"):
            st.image("assets/image_logo-modified.png", width=400)
        elif os.path.exists("assets/image_logo.jpg"):
            st.image("assets/image_logo.jpg", width=400)
        else:
            st.info("(Logo image not found in assets/)")

                        
    st.markdown("""
    <p style="text-align:center; font-size:22px; color: #777; margin-top:20px;">
    Where detectives uncover hidden moments inside videos
    </p>
    """, unsafe_allow_html=True)

                       
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        clicked = st.button("🔎 Begin the Investigation", type="primary", use_container_width=True)
        if clicked:
            st.session_state.page = "search"
            st.rerun()

                              
                                                                               
def render_search_page():
                            
    if st.sidebar.button("← Back to Home"):
        st.session_state.page = "home"
        st.rerun()

    st.title("🎬 ClipCam (Text-to-Video Moment Search)")

                     
    with st.sidebar:
        st.subheader("Options")
        model_name = st.text_input("Model", MODEL_NAME)
        pretrained = st.text_input("Pretrained", PRETRAINED)
        sample_fps = st.number_input("Sample FPS (for indexing)", value=float(SAMPLE_FPS), min_value=0.5, step=0.5)
        windows_default_str = ", ".join(str(w) for w in WINDOWS_DEFAULT)
        windows_str = st.text_input("Window sizes (sec, comma-separated)", windows_default_str)
        min_gap_s = st.number_input("Min gap between picks (sec)", value=float(MIN_GAP_SEC), min_value=0.0, step=0.5)
        
                             
        reencode = st.checkbox("Re-encode exported clip", value=False)
        save_keyframes_flag = st.checkbox("Also save first/last frame", value=False)
        draw_boxes = st.checkbox("Draw Bounding Boxes", value=True)

                 
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv", "avi", "webm"])
    query = st.text_input("Prompt / Query", placeholder="e.g., 'a dog jumping into the pool'")

    run = st.button("Find Evidence")

                          
    if run:
        if not video_file:
            st.error("Please upload a video.")
            st.stop()
        if not query.strip():
            st.error("Please enter a query.")
            st.stop()
            
                       
        try:
            windows_s = [float(x.strip()) for x in windows_str.split(",") if x.strip()]
            if not windows_s: raise ValueError
        except Exception:
            st.error("Window sizes must be a comma-separated list of numbers (seconds).")
            st.stop()

                                 
        detector_instance = None
        if draw_boxes:
            with st.spinner("Loading Object Detection Model..."):
                detector_instance = load_detector()

                                  
        st.session_state.chat_history = []
        st.session_state.gemini_instance = None
        st.session_state.current_clip_path = None

        with st.spinner("Indexing and searching…"):
            video_path = ensure_video_on_disk(video_file)

            out = run_search_top1(
                video_path=video_path,
                query=query.strip(),
                windows_s=windows_s,
                min_gap_s=min_gap_s,
                sample_fps=sample_fps,
                model_name=model_name,
                pretrained=pretrained,
                reencode=reencode,
                save_keyframes_flag=save_keyframes_flag,
                draw_boxes=draw_boxes,
                detector=detector_instance
            )

        results = out["results"]
        if not results:
            st.warning("No segments found.")
            st.stop()
        
        if out["clip_path"] and out["clip_path"].exists():
            st.session_state.current_clip_path = out["clip_path"]
        else:
            st.error("Could not extract clip.")
            st.stop()

                                               
        with st.spinner("Analyzing clip..."):
            gemini = GeminiChat()
            gemini.upload_video(st.session_state.current_clip_path)
            initial_response = gemini.start_chat(query)
            
            st.session_state.gemini_instance = gemini
            st.session_state.chat_history.append({"role": "assistant", "content": initial_response})

                         
        t0, t1, sc, W = results[0]
        st.success(f"Top segment: {t0:.2f}s → {t1:.2f}s  (W={W}s)")

                                                     
    if st.session_state.current_clip_path:
        st.subheader("Best Match")
        st.video(str(st.session_state.current_clip_path))
        
                                                                  
        with open(st.session_state.current_clip_path, "rb") as f:
            st.download_button(
                label="Download Clip",
                data=f, 
                file_name=st.session_state.current_clip_path.name,
                mime="video/mp4"
            )

    if st.session_state.gemini_instance:
        st.markdown("---")
        st.subheader("Chat about this incident")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_input := st.chat_input("Ask a follow-up question..."):
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_text = st.session_state.gemini_instance.send_message(user_input)
                    st.markdown(response_text)
            
            st.session_state.chat_history.append({"role": "assistant", "content": response_text})

inject_custom_css()
                          
if "page" not in st.session_state:
    st.session_state.page = "home"
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "gemini_instance" not in st.session_state:
    st.session_state.gemini_instance = None
if "current_clip_path" not in st.session_state:
    st.session_state.current_clip_path = None

                      
if st.session_state.page == "search":
    render_search_page()
else:
    render_home_page()