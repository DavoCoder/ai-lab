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

# summarization_post_processor.py
from transformers import pipeline
from response_post_processing.response_post_processor_interface import ResponsePostProcessor

class SummarizationPostProcessor(ResponsePostProcessor):
    """
    Summarizes verbose responses into concise summaries.
    """

    def __init__(self, model_name="t5-small"):
        self.summarizer = pipeline("summarization", model=model_name)

    def process(self, response: str, retrieved_docs: list) -> str:
        """
        Summarizes the response.

        Args:
            response (str): Generated response.
            retrieved_docs (list): Retrieved documents (unused here).

        Returns:
            str: Summarized response.
        """
        summary = self.summarizer(response, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
        print("Response summarized successfully.")
        return summary
