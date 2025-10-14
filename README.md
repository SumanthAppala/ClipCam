  # ClipCAM

Type a phrase → jump to the right part of a long video. This repo uses CLIP (ViT-B/32) to index frames at 1 fps, then performs sliding-window text–image matching to return time-stamped segments. Outputs (CSV, thumbnails, optional MP4 clips) live in `results/`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```
Put videos in data/ (or pass an absolute/relative path).

## Usage
1) Build index (once per video)
```bash
python app.py index --video data/my_video.mp4
# or shorthand searches data/: python app.py index --video my_video.mp4
```
2) Search moments
```bash
python app.py search --video my_video.mp4 \
  --query "person opens the door" \
  --windows 3 5 10 --topk 5 \
  --save-clips --clip-method ffmpeg  # optional MP4 cutouts
```

## Outputs (results/<video_stem>/)

- index_*.npz — cached CLIP frame embeddings

- results_<query>.csv — start_sec, end_sec, score, window_s

- thumbs/ — JPEG thumbnails for top segments

- clips/ — MP4 segments (if --save-clips)

## Notes

- Defaults use ViT-B/32 (openai). Change via --model --pretrained.

- For frame-accurate MP4s with audio, use --clip-method ffmpeg --reencode.

- Index sampling uses --sample-fps (default 1.0); higher fps = better recall, larger index.