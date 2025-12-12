from pathlib import Path
import torch

                 
MODEL_NAME = "ViT-B-32"
PRETRAINED = "openai"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

                   
SAMPLE_FPS = 1.0                                                         
BATCH_SIZE = 64                                          
WINDOWS_DEFAULT = [3, 5, 10]            
MIN_GAP_SEC = 2.0                                                  

    
DATA_ROOT = Path("data")
RESULTS_ROOT = Path("results")

                                                                       
INDEX_NAME_FMT = "index_{model}_{pretrained}.npz"
THUMBS_DIRNAME = "thumbs"
CLIPS_DIRNAME = "clips"
RESULTS_CSV_FMT = "results_{slug}.csv"
