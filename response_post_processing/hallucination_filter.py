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

# hallucination_filter.py
from response_post_processing.response_post_processor_interface import ResponsePostProcessor

class HallucinationFilter(ResponsePostProcessor):
    """
    Filters hallucinated content by checking overlap with retrieved documents.
    """

    def process(self, response: str, retrieved_docs: list) -> str:
        """
        Flags or removes hallucinated content.

        Args:
            response (str): Generated response.
            retrieved_docs (list): Retrieved documents.

        Returns:
            str: Filtered response or a warning.
        """
        for doc in retrieved_docs:
            if doc.page_content in response:
                return response  # Safe response with overlap

        print("Hallucination detected! Generated response may not match retrieved content.")
        return "I'm sorry, I could not generate a factually grounded response."
