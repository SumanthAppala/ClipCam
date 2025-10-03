from pathlib import Path
import torch

# Model / runtime
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Sampling & search
SAMPLE_FPS = 1.0               # frames per second to sample for indexing
BATCH_SIZE = 64                # CLIP encoding batch size
WINDOWS_DEFAULT = [3, 5, 10]   # seconds
MIN_GAP_SEC = 2.0              # min separation between picks (sec)

# IO
DATA_ROOT = Path("data")
RESULTS_ROOT = Path("results")

# Filenames / dirs (per-video subdir inside RESULTS_ROOT/<video_stem>/)
INDEX_NAME_FMT = "index_{model}_{pretrained}.npz"
THUMBS_DIRNAME = "thumbs"
CLIPS_DIRNAME = "clips"
RESULTS_CSV_FMT = "results_{slug}.csv"
