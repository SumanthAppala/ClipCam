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

## How it works

flowchart TD
  A[User] --> B{Open app}
  B --> C[Streamlit UI]
  B --> D[CLI]
  C --> E{Mode?}
  D --> E
  E -->|Index| I1[Read meta]
  I1 --> I2[Timestamps]
  I2 --> I3[Grab frames]
  I3 --> I4[CLIP encode]
  I4 --> I5[Save NPZ]
  I5 --> Z[Ready]
  E -->|Search| S1[Ensure index]
  S1 --> S2[Encode query]
  S2 --> S3[Score frames]
  S3 --> S4{Mode}
  S4 -->|Windows| S5[Slide and mean]
  S4 -->|Auto| S7[Kadane]
  S5 --> S8[Pick segments]
  S7 --> S8
  S8 --> S9[CSV]
  S8 --> S10[Thumbs]
  S8 --> S11{Clips?}
  S11 -->|Yes| S12[Export MP4]
  S11 -->|No| S14[Skip]
  S12 --> S13{Edge frames?}
  S13 -->|Yes| S15[Save first/last jpg]
  S13 -->|No| S16[Skip]
  C --> U1[Streamlit controls]
  U1 --> U2[Run search]
  U2 --> U3[Show results]

