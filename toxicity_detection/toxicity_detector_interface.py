from abc import ABC, abstractmethod

class ToxicityDetector(ABC):
    """
    Abstract class for detecting toxic content.
    """

    @abstractmethod
    def detect_toxicity(self, text: str) -> bool:
        """
        Detect if the given text contains toxic content.

        Args:
            text (str): The input text to analyze.

        Returns:
            bool: True if toxic content is detected, otherwise False.
        """
        pass
