from transformers import pipeline
from toxicity_detection.toxicity_detector_interface import ToxicityDetector

class HuggingFaceToxicityDetector(ToxicityDetector):
    """
    Toxicity detector using HuggingFace's 'toxicity' model.
    """

    def __init__(self, model_name="unitary/toxic-bert"):
        self.pipeline = pipeline("text-classification", model=model_name)

    def detect_toxicity(self, text: str) -> bool:
        result = self.pipeline(text)
        for res in result:
            if res["label"] == "toxic" and res["score"] > 0.6:
                print("HuggingFace: Toxic content detected.")
                return True
        return False
