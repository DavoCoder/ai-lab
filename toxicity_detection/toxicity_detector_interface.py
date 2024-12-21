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

# toxicity_detector_interface.py
from abc import ABC, abstractmethod

class ToxicityDetector(ABC):
    """
    Abstract class for detecting toxic content.
    """

    @abstractmethod
    def detect_toxicity(self, text: str) -> bool:
        """
        Detect if the given text contains toxic content.

        Args:
            text (str): The input text to analyze.

        Returns:
            bool: True if toxic content is detected, otherwise False.
        """
