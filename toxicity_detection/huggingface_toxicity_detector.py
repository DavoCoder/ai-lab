
# Copyright 2024-2025 DavoCoder
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# huggingface_toxicity_detector.py
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
