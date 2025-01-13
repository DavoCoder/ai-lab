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

# embedding_model_interface.py
from abc import ABC, abstractmethod

class EmbeddingModel(ABC):
    @abstractmethod
    def load_model(self):
        """
        Load the embedding model.
        Returns:
            An initialized embedding model.
        """

    @abstractmethod
    def embed_query(self, query: str):
        """
        Embed a query.
        Args:
            query: The query to embed.
        Returns:
            An embedding vector.
        """
