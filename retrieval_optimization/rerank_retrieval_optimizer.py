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

# rerank_retrieval_optimizer.py
from transformers import pipeline
from retrieval_optimization.retrieval_optimizer_interface import RetrievalOptimizer

class ReRankRetrievalOptimizer(RetrievalOptimizer):
    """
    Re-ranks retrieved documents using a cross-encoder model for better accuracy.
    """

    def __init__(self, model_name="cross-encoder/ms-marco-MiniLM-L6"):
        self.reranker = pipeline("text-classification", model=model_name)

    def optimize(self, query: str, retrieved_docs: list) -> list:
        """
        Re-ranks the retrieved documents based on their relevance to the query.

        Args:
            query (str): User query.
            retrieved_docs (list): Retrieved documents.

        Returns:
            list: Re-ranked list of documents.
        """
        scored_docs = []
        for doc in retrieved_docs:
            score = self.reranker({"text": doc.page_content, "query": query})[0]['score']
            scored_docs.append((score, doc))

        # Sort by relevance score
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        print("ReRanker: Retrieved documents have been re-ranked.")
        return [doc for _, doc in scored_docs]
