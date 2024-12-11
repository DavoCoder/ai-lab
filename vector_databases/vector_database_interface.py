from abc import ABC, abstractmethod

class VectorDatabase(ABC):
    @abstractmethod
    def load_or_initialize(self, documents=None):
        """
        Load an existing vector database or initialize a new one.
        Args:
            documents (list): Documents to add during initialization.
        """
        pass

    @abstractmethod
    def add_documents(self, documents):
        """
        Add documents to the vector database.
        Args:
            documents (list): List of documents to add.
        """
        pass

    @abstractmethod
    def delete_documents(self, document_ids):
        """
        Delete documents from the vector database.
        Args:
            document_ids (list): List of document IDs to delete.
        """
        pass

    @abstractmethod
    def get_retriever(self, k=3):
        """
        Return a retriever for similarity search.
        Args:
            k (int): Number of top results to retrieve.
        """
        pass