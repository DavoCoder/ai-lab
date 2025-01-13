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

# openai_embedding_model.py
import os
from langchain_openai import OpenAIEmbeddings
from embeddings.embedding_model_interface import EmbeddingModel

class OpenAIEmbeddingModel(EmbeddingModel):
    """
    OpenAI Embedding Model Wrapper.
    """
    def __init__(self, api_key):
        """
        OpenAI Embedding Model Wrapper.
        
        Args:
            api_key (str): OpenAI API key.
        """
        self.api_key = api_key
        self.model = None
    def load_model(self):
        """
        Load the OpenAI Embeddings model.
        Returns:
            OpenAIEmbeddings instance.
        """
        os.environ["OPENAI_API_KEY"] = self.api_key
        print("OpenAI Embeddings model loaded successfully.")
        self.model = OpenAIEmbeddings()
        return self.model
    
    def embed_query(self, query: str):
        """
        Embed a query.
        Args:
            query: The query to embed.
        Returns:
            An embedding vector.
        """
        if self.model is None:
            self.load_model()
        return self.model.embed_query(query)
