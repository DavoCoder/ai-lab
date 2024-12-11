from spellchecker import SpellChecker
from query_pre_processing.query_processor_interface import QueryProcessor

class SpellCheckQueryProcessor(QueryProcessor):
    """
    Query processor for spell checking.
    """

    def __init__(self):
        self.spellchecker = SpellChecker()

    def process(self, query: str) -> str:
        corrected_query = " ".join([self.spellchecker.correction(word) for word in query.split()])
        print(f"Original Query: {query} | Corrected Query: {corrected_query}")
        return corrected_query
