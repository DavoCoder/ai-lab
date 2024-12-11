from response_post_processing.response_post_processor_interface import ResponsePostProcessor

class HallucinationFilter(ResponsePostProcessor):
    """
    Filters hallucinated content by checking overlap with retrieved documents.
    """

    def process(self, response: str, retrieved_docs: list) -> str:
        """
        Flags or removes hallucinated content.

        Args:
            response (str): Generated response.
            retrieved_docs (list): Retrieved documents.

        Returns:
            str: Filtered response or a warning.
        """
        for doc in retrieved_docs:
            if doc.page_content in response:
                return response  # Safe response with overlap

        print("Hallucination detected! Generated response may not match retrieved content.")
        return "I'm sorry, I could not generate a factually grounded response."
