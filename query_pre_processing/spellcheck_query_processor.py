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

# spellcheck_query_processor.py
from spellchecker import SpellChecker
from query_pre_processing.query_processor_interface import QueryProcessor

class SpellCheckQueryProcessor(QueryProcessor):
    """
    Query processor for spell checking.
    """

    def __init__(self):
        self.spellchecker = SpellChecker()

    def process(self, query: str) -> str:
        corrected_query = " ".join([self.spellchecker.correction(word) for word in query.split()])
        print(f"Original Query: {query} | Corrected Query: {corrected_query}")
        return corrected_query
