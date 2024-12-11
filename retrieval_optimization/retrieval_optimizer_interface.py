from abc import ABC, abstractmethod

class RetrievalOptimizer(ABC):
    """
    Abstract class for optimizing the retrieval process.
    """

    @abstractmethod
    def optimize(self, query: str, retrieved_docs: list) -> list:
        """
        Optimize the retrieved documents.

        Args:
            query (str): The search query.
            retrieved_docs (list): A list of retrieved documents.

        Returns:
            list: A list of optimized/re-ranked documents.
        """
        pass
