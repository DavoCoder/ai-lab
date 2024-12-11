from abc import ABC, abstractmethod

class ResponsePostProcessor(ABC):
    """
    Abstract class for post-processing the LLM-generated response.
    """

    @abstractmethod
    def process(self, response: str, retrieved_docs: list) -> str:
        """
        Process the LLM-generated response.

        Args:
            response (str): The raw response generated by the LLM.
            retrieved_docs (list): A list of documents used for generating the response.

        Returns:
            str: The processed response.
        """
        pass
