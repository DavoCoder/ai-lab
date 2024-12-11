from abc import ABC, abstractmethod

class QueryProcessor(ABC):
    """
    Abstract class for query preprocessing and augmentation.
    """

    @abstractmethod
    def process(self, query: str) -> str:
        """
        Process or augment the query.

        Args:
            query (str): The original user query.

        Returns:
            str: The processed or augmented query.
        """
        pass
