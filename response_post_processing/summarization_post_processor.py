from transformers import pipeline
from response_post_processing.response_post_processor_interface import ResponsePostProcessor

class SummarizationPostProcessor(ResponsePostProcessor):
    """
    Summarizes verbose responses into concise summaries.
    """

    def __init__(self, model_name="t5-small"):
        self.summarizer = pipeline("summarization", model=model_name)

    def process(self, response: str, retrieved_docs: list) -> str:
        """
        Summarizes the response.

        Args:
            response (str): Generated response.
            retrieved_docs (list): Retrieved documents (unused here).

        Returns:
            str: Summarized response.
        """
        summary = self.summarizer(response, max_length=50, min_length=10, do_sample=False)[0]['summary_text']
        print("Response summarized successfully.")
        return summary
