# pages/clipcam_search.py
import streamlit as st
from pathlib import Path
from PIL import Image

from utils.common import ensure_dir
from utils.indexer import build_index_npz
from utils.encoder import ClipEncoder
from utils.search import per_frame_scores, pick_top_segments
from utils.export import export_clips
from utils.qna import refine_query, answer_question


def main():
    # --- Page setup ---
    st.set_page_config(page_title="CLIP Video Moment Search", layout="wide")
    st.title("🎬 CLIP Video Moment Search")

    # --- Video upload ---
    st.header("Upload Videos")
    uploaded_files = st.file_uploader("Upload videos", type=["mp4", "webm"], accept_multiple_files=True)

    video_paths = []
    if uploaded_files:
        data_dir = Path("data")
        ensure_dir(data_dir)
        for file in uploaded_files:
            video_path = data_dir / file.name
            with open(video_path, "wb") as f:
                f.write(file.getbuffer())
            video_paths.append(video_path)

    # --- Query input ---
    st.header("Search Query")
    query = st.text_input("Enter your text query (e.g., 'man cycling')")

    # --- Parameters ---
    st.sidebar.header("Parameters")
    model_name = st.sidebar.selectbox("Model", ["ViT-B-32", "RN50x16"])
    pretrained = st.sidebar.selectbox("Pretrained", ["openai", "laion2b_s34b_b79k"])
    sample_fps = st.sidebar.number_input("Sample FPS", min_value=0.1, max_value=30.0, value=2.0, step=0.1)
    windows_input = st.sidebar.text_input("Window sizes (seconds, comma-separated)", "4,8,12")
    min_gap = st.sidebar.number_input("Min gap between picks (seconds)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    clip_method = st.sidebar.selectbox("Clip export method", ["auto", "ffmpeg", "opencv"])
    reencode = st.sidebar.checkbox("Reencode clips", value=False)
    topk = st.sidebar.number_input("Number of clips", min_value=1, max_value=10, value=3)

    # Convert windows input to list of floats
    try:
        windows = [float(w.strip()) for w in windows_input.split(",")]
    except:
        windows = [4, 8, 12]

    # --- Run search ---
    if st.button("Search Moments"):
        if not video_paths:
            st.warning("Please upload at least one video.")
        elif not query:
            st.warning("Please enter a search query.")
        else:
            for video_path in video_paths:
                st.info(f"Processing video: {video_path.name}")

                # Step 1: Index video
                st.write("Indexing video...")
                out_dir = Path("results") / video_path.stem
                idx_path = build_index_npz(video_path, out_dir, sample_fps, model_name, pretrained)

                # Step 2: Encode query
                st.write("Encoding query...")
                encoder = ClipEncoder(model_name=model_name, pretrained=pretrained)
                query_vec = encoder.encode_text(query)

                # Step 3: Load video embeddings
                from utils.indexer import load_index_npz
                E, ts, meta = load_index_npz(idx_path)

                # Step 4: Compute per-frame similarity
                scores = per_frame_scores(E, query_vec)

                # Step 5: Pick top segments
                segments = pick_top_segments(scores, windows_frames=[int(w*sample_fps) for w in windows],
                                             topk=topk,
                                             min_gap_frames=int(min_gap*sample_fps))

                # Step 6: Export clips
                st.write("Exporting clips...")
                clip_dir = out_dir / "clips"
                export_clips(video_path, segments, clip_dir, method=clip_method, reencode=reencode)

                # Step 7: Display clips
                st.subheader(f"Generated Clips for {video_path.name}")
                if clip_dir.exists():
                    for clip_file in sorted(clip_dir.glob("*.mp4")):
                        st.video(str(clip_file))
                else:
                    st.warning("No clips generated for this video.")

'''
# pages/clipcam_search.py
import streamlit as st
from pathlib import Path
from PIL import Image

from utils.common import ensure_dir
from utils.indexer import build_index_npz, load_index_npz
from utils.encoder import ClipEncoder
from utils.search import per_frame_scores, pick_top_segments
from utils.export import export_clips
from utils.qna import refine_query, answer_question


def main():
    # --- Page setup ---
    st.set_page_config(page_title="CLIP Video Moment Search", layout="wide")
    st.title("🎬 CLIP Video Moment Search")

    # --- Video upload ---
    st.header("Upload Videos")
    uploaded_files = st.file_uploader(
        "Upload videos", type=["mp4", "webm"], accept_multiple_files=True
    )

    video_paths = []
    if uploaded_files:
        data_dir = Path("data")
        ensure_dir(data_dir)
        for file in uploaded_files:
            video_path = data_dir / file.name
            with open(video_path, "wb") as f:
                f.write(file.getbuffer())
            video_paths.append(video_path)

    # --- Query input ---
    st.header("Search Query")
    query = st.text_input("Enter your text query (e.g., 'man cssycling')", key="main_query")

    # Automatically refine query using AI
    if query:
        if "refined_query" not in st.session_state or st.session_state.get("last_query") != query:
            st.session_state.refined_query = refine_query(query)
            st.session_state.last_query = query

    refined_query = st.session_state.get("refined_query", query)
    st.write(f"**Refined Query:** {refined_query}")

    # --- Parameters ---
    st.sidebar.header("Parameters")
    model_name = st.sidebar.selectbox("Model", ["ViT-B-32", "RN50x16"])
    pretrained = st.sidebar.selectbox("Pretrained", ["openai", "laion2b_s34b_b79k"])
    sample_fps = st.sidebar.number_input("Sample FPS", min_value=0.1, max_value=30.0, value=2.0, step=0.1)
    windows_input = st.sidebar.text_input("Window sizes (seconds, comma-separated)", "4,8,12")
    min_gap = st.sidebar.number_input("Min gap between picks (seconds)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    clip_method = st.sidebar.selectbox("Clip export method", ["auto", "ffmpeg", "opencv"])
    reencode = st.sidebar.checkbox("Reencode clips", value=False)
    topk = st.sidebar.number_input("Number of clips", min_value=1, max_value=10, value=3)

    # Convert windows input to list of floats
    try:
        windows = [float(w.strip()) for w in windows_input.split(",")]
    except:
        windows = [4, 8, 12]

    # --- Run search ---
    if st.button("Search Moments"):
        if not video_paths:
            st.warning("Please upload at least one video.")
        elif not refined_query:
            st.warning("Please enter a search query.")
        else:
            for video_path in video_paths:
                st.info(f"Processing video: {video_path.name}")

                # Step 1: Index video
                st.write("Indexing video...")
                out_dir = Path("results") / video_path.stem
                idx_path = build_index_npz(video_path, out_dir, sample_fps, model_name, pretrained)

                # Step 2: Encode query
                st.write("Encoding query...")
                encoder = ClipEncoder(model_name=model_name, pretrained=pretrained)
                query_vec = encoder.encode_text(refined_query)

                # Step 3: Load video embeddings
                E, ts, meta = load_index_npz(idx_path)

                # Step 4: Compute per-frame similarity
                scores = per_frame_scores(E, query_vec)

                # Step 5: Pick top segments
                segments = pick_top_segments(
                    scores,
                    windows_frames=[int(w * sample_fps) for w in windows],
                    topk=topk,
                    min_gap_frames=int(min_gap * sample_fps)
                )

                # Step 6: Export clips
                st.write("Exporting clips...")
                clip_dir = out_dir / "clips"
                export_clips(video_path, segments, clip_dir, method=clip_method, reencode=reencode)

                # Step 7: Display clips
                st.subheader(f"Generated Clips for {video_path.name}")
                if clip_dir.exists():
                    for clip_file in sorted(clip_dir.glob("*.mp4")):
                        st.video(str(clip_file))
                else:
                    st.warning("No clips generated for this video.")

                # --- Conversational Follow-Ups ---
                st.subheader("💬 Ask a follow-up question about the results")

                if "chat_history" not in st.session_state:
                    st.session_state.chat_history = []

                with st.form(key="chat_form"):
                    user_question = st.text_input(
                        "Your question (e.g., 'What color is his shirt?')",
                        key="chat_input"
                    )
                    submit_button = st.form_submit_button("Ask")

                if submit_button and user_question:
                    video_summary = f"User searched for: {refined_query}. Extracted segments: {segments}"
                    answer = answer_question(
                        st.session_state.chat_history,
                        user_question,
                        video_summary
                    )
                    st.session_state.chat_history.append({"role": "user", "content": user_question})
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})

                # Display chat history
                for msg in st.session_state.chat_history:
                    role = "User" if msg["role"] == "user" else "AI"
                    st.write(f"**{role}:** {msg['content']}")
'''