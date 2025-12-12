Here is the complete `README.md` file with **Step 2 (API Keys)** and everything following it (Usage, Outputs, Configuration) fully integrated into the Markdown code block.

````markdown
# ClipCAM: Semantic Video Search & Conversational AI

**ClipCAM** is a multi-modal computer vision tool that allows users to "talk" to their videos. By leveraging **CLIP (ViT-B/32)** for semantic indexing and **Google Gemini** for conversational reasoning, this tool enables users to find specific moments in long videos using natural language queries and engage in a dialogue about the video content.

### Key Features
* **Semantic Search:** Type a natural language phrase (e.g., *"person opens the door"*) to find exact timestamps.
* **Conversational Interface:** Integrated with **Google Gemini** to answer questions about the video content based on indexed metadata.
* **Dual Interface:** Run as a robust Command Line Interface (CLI) for batch processing or a **Streamlit Web App** for interactive exploration.
* **Multi-Modal Output:** Generates CSV logs, thumbnails, and clipped MP4 segments.

---

## 🛠 Setup

### 1. Environment
Create a virtual environment to keep dependencies isolated.

```bash
# Create virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate      # Mac/Linux
# .venv\Scripts\activate       # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
````

### 2\. API Keys (Required for Chat)

To use the conversational features (Gemini), you must set your Google API key.

**Mac/Linux:**

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

**Windows (PowerShell):**

```powershell
$env:GOOGLE_API_KEY="your_api_key_here"
```

-----

## 🚀 Usage

You can use ClipCAM in two modes: **Interactive Web App** or **CLI**.

### A. Interactive Web App (Streamlit)

Launch the GUI to upload videos, visualize search results, and chat with the AI assistant.

```bash
streamlit run app.py
```

  * **Visual Search:** View top-k matching frames instantly.
  * **Chat:** Ask Gemini questions like *"Summarize the events in the video"* or *"What color was the car in the second clip?"*

### B. Command Line Interface (CLI)

#### 1\. Build Index (Pre-processing)

Extracts frames (default 1 fps) and computes CLIP embeddings. This must be done once per video.

```bash
python app.py index --video data/my_video.mp4

# Note: The tool automatically checks the 'data/' folder.
# Shorthand: python app.py index --video my_video.mp4
```

#### 2\. Search Moments

Perform sliding-window text–image matching to return time-stamped segments.

```bash
python app.py search --video my_video.mp4 \
  --query "person opens the door" \
  --windows 3 5 10 --topk 5 \
  --save-clips --clip-method ffmpeg
```

-----

## 📂 Outputs

Results are saved in `results/<video_stem>/`:

| File/Folder | Description |
| :--- | :--- |
| `index_*.npz` | Cached CLIP frame embeddings (re-used for faster searches). |
| `results_<query>.csv` | Detailed logs containing `start_sec`, `end_sec`, similarity scores, and window sizes. |
| `thumbs/` | JPEG thumbnails of the highest-scoring frames. |
| `clips/` | Exact MP4 cutouts (generated if `--save-clips` is used). |

-----

## ⚙️ Configuration & Advanced Options

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--model` | `ViT-B/32` | The CLIP model architecture to use. |
| `--pretrained` | `openai` | The pre-trained weights source. |
| `--sample-fps` | `1.0` | Frames per second to index. Higher FPS = better recall but larger index files. |
| `--clip-method` | `ffmpeg` | Use `ffmpeg` for precise cutting or `opencv` for faster, less accurate cutting. |
| `--reencode` | `False` | If using ffmpeg, this forces re-encoding for frame-accurate cuts (slower). |

-----

## 📜 Project Context

This project was developed as a Computer Vision final project (University of Pennsylvania), exploring the intersection of contrastive language-image pre-training (CLIP) and Large Language Models (LLMs) for video understanding.

```
```