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

# openai_toxicity_detector.py
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
