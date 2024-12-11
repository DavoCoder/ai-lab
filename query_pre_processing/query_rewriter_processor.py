from transformers import pipeline
from query_pre_processing.query_processor_interface import QueryProcessor

class QueryRewriterProcessor(QueryProcessor):
    """
    Query processor for rewriting/expanding queries using a language model.
    """

    def __init__(self, model_name="t5-small"):
        self.rewriter = pipeline("text2text-generation", model=model_name)

    def process(self, query: str) -> str:
        rewritten = self.rewriter(f"Rewrite this query: {query}", max_length=50, num_return_sequences=1)
        rewritten_query = rewritten[0]["generated_text"]
        print(f"Original Query: {query} | Rewritten Query: {rewritten_query}")
        return rewritten_query
