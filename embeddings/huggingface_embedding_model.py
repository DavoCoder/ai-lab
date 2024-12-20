# Copyright 2024-2025 DavoCoder
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# huggingface_embedding_model.py
from langchain_huggingface import HuggingFaceEmbeddings
from embeddings.embedding_model_interface import EmbeddingModel

class HuggingFaceEmbeddingModel(EmbeddingModel):
    """
    HuggingFace Embedding Model Wrapper.
    """
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
