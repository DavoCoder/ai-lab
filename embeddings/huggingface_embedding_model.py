from langchain_huggingface import HuggingFaceEmbeddings
from embeddings.embedding_model_interface import EmbeddingModel

class HuggingFaceEmbeddingModel(EmbeddingModel):
    
    HUGGINGFACE_MODELS = {
        "all-mpnet-base-v2": 768,
        "all-MiniLM-L6-v2": 384,
        "all-MiniLM-L12-v2": 384,
        "paraphrase-multilingual-mpnet-base-v2": 768,
        "sentence-t5-xxl": 1024  
    }

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        HuggingFace Embedding Model Wrapper.
        
        Args:
            model_name (str): Name of the HuggingFace model to load.
        """
        self.model_name = model_name

    def load_model(self):
        """
        Load the HuggingFace Embeddings model.
        Returns:
            HuggingFaceEmbeddings instance.
        """
        print(f"Loading HuggingFace model: {self.model_name}")
        return HuggingFaceEmbeddings(model_name=self.model_name)
