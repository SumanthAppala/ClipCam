from typing import List
import numpy as np
import torch
import open_clip
from PIL import Image

from config import MODEL_NAME, PRETRAINED, DEVICE, BATCH_SIZE
from .common import chunks

class ClipEncoder:
    def __init__(self, model_name: str = MODEL_NAME, pretrained: str = PRETRAINED, device: str = DEVICE):
        self.device = device
        model, _preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = model.eval().to(device)
        self.preprocess = preprocess_val
        self.tokenizer = open_clip.get_tokenizer(model_name)

    @torch.inference_mode()
    def encode_images(self, images: List[Image.Image]) -> np.ndarray:
        if not images:
            # 512 is typical for ViT-B/32; not used when empty
            return np.zeros((0, 512), dtype=np.float32)
        outs = []
        for batch in chunks(images, BATCH_SIZE):
            ims = torch.stack([self.preprocess(im) for im in batch]).to(self.device)
            feats = self.model.encode_image(ims)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            outs.append(feats.detach().cpu().numpy().astype("float32"))
        return np.concatenate(outs, axis=0)

    @torch.inference_mode()
    def encode_text(self, text: str) -> np.ndarray:
        t = self.tokenizer([text])
        te = self.model.encode_text(t.to(self.device))
        te = (te / te.norm(dim=-1, keepdim=True)).detach().cpu().numpy().astype("float32")
        return te[0]
