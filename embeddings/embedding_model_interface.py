from abc import ABC, abstractmethod

class EmbeddingModel(ABC):
    @abstractmethod
    def load_model(self):
        """
        Load the embedding model.
        Returns:
            An initialized embedding model.
        """
        pass
