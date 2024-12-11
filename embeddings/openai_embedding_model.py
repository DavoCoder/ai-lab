import os
from langchain_community.embeddings import OpenAIEmbeddings
from embeddings.embedding_model_interface import EmbeddingModel

class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, api_key):
        """
        OpenAI Embedding Model Wrapper.
        
        Args:
            api_key (str): OpenAI API key.
        """
        self.api_key = api_key

    def load_model(self):
        """
        Load the OpenAI Embeddings model.
        Returns:
            OpenAIEmbeddings instance.
        """
        os.environ["OPENAI_API_KEY"] = self.api_key
        print("OpenAI Embeddings model loaded successfully.")
        return OpenAIEmbeddings()
