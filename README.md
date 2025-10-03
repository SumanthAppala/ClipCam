# ClipCAM

Type a phrase → jump to the right part of a long video. This repo uses CLIP (ViT-B/32) to index frames at 1 fps, then performs sliding-window text–image matching to return time-stamped segments. Outputs (CSV, thumbnails, optional MP4 clips) live in `results/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
