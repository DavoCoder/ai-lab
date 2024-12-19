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

# query_rewriter_processor.py
from transformers import pipeline
from query_pre_processing.query_processor_interface import QueryProcessor

class QueryRewriterProcessor(QueryProcessor):
    """
    Query processor for rewriting/expanding queries using a language model.
    """

    def __init__(self, model_name="t5-small"):
        self.rewriter = pipeline("text2text-generation", model=model_name)

    def process(self, query: str) -> str:
        rewritten = self.rewriter(f"Rewrite this query: {query}", max_length=50, num_return_sequences=1)
        rewritten_query = rewritten[0]["generated_text"]
        print(f"Original Query: {query} | Rewritten Query: {rewritten_query}")
        return rewritten_query
