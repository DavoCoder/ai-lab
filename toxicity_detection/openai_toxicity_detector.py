from openai import OpenAI
from toxicity_detection.toxicity_detector_interface import ToxicityDetector

class OpenAIToxicityDetector(ToxicityDetector):
    """
    Toxicity detector using OpenAI's moderation endpoint.
    """

    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def detect_toxicity(self, text: str) -> bool:
        response = self.client.moderations.create(input=text)
        response_dict = response.model_dump()

        if response_dict["results"][0]["flagged"]:
            print("OpenAI: Toxic content detected.")
            return True
        return False
