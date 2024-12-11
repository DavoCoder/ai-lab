from abc import ABC, abstractmethod

class ExternalKnowledgeBase(ABC):
    @abstractmethod
    def connect(self):
        """
        Establish a connection to the external knowledge base.
        """
        pass

    @abstractmethod
    def fetch_documents(self):
        """
        Fetch or query documents from the external knowledge base.
        Returns:
            List of documents.
        """
        pass

    @abstractmethod
    def detect_updates(self):
        """
        Detect new or updated documents in the external knowledge base.
        Returns:
            List of new or updated documents.
        """
        pass
