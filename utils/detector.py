import torch
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

class ObjectDetector:
    def __init__(self, device="cpu"):
        self.device = device
        print(f"[Detector] Loading OWL-ViT on {device}...")
        self.processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch32")
        self.model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch32").to(self.device)
        self.model.eval()

    def detect(self, image: Image.Image, text_queries: list, threshold=0.1):
        """
        Returns a list of [x_min, y_min, x_max, y_max, score, label]
        """
        inputs = self.processor(text=text_queries, images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

                                           
        target_sizes = torch.Tensor([image.size[::-1]]).to(self.device)
        
                                                                       
        results = self.processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=threshold
        )[0]

        detections = []
        for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
                                             
            box = [round(i, 2) for i in box.tolist()]
            detections.append({
                "box": box,
                "score": round(score.item(), 3),
                "label": text_queries[label]
            })
        
        return detections