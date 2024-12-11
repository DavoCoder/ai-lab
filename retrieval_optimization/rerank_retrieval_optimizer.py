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
