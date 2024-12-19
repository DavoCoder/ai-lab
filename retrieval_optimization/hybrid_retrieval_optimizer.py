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

# hybrid_retrieval_optimizer.py
from retrieval_optimization.retrieval_optimizer_interface import RetrievalOptimizer

class HybridRetrievalOptimizer(RetrievalOptimizer):
    """
    Combines vector search with keyword-based search for hybrid retrieval.
    """

    def __init__(self, keyword_retriever, vector_retriever):
        """
        Args:
            keyword_retriever: A keyword-based retriever.
            vector_retriever: A vector-based retriever.
        """
        self.keyword_retriever = keyword_retriever
        self.vector_retriever = vector_retriever

    def optimize(self, query: str, retrieved_docs: list) -> list:
        """
        Combines results from keyword and vector retrievers.

        Args:
            query (str): User query.
            retrieved_docs (list): Initial vector-retrieved documents.

        Returns:
            list: Combined and deduplicated documents.
        """
        keyword_results = self.keyword_retriever.get_relevant_documents(query)
        combined_results = {doc.page_content: doc for doc in retrieved_docs}

        for doc in keyword_results:
            combined_results[doc.page_content] = doc

        print(f"Hybrid Retrieval: Combined {len(combined_results)} unique results.")
        return list(combined_results.values())
