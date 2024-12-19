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

# vector_database_interface.py
from abc import ABC, abstractmethod

class VectorDatabase(ABC):
    @abstractmethod
    def load_or_initialize(self, documents=None):
        """
        Load an existing vector database or initialize a new one.
        Args:
            documents (list): Documents to add during initialization.
        """
        pass

    @abstractmethod
    def add_documents(self, documents):
        """
        Add documents to the vector database.
        Args:
            documents (list): List of documents to add.
        """
        pass

    @abstractmethod
    def delete_documents(self, document_ids):
        """
        Delete documents from the vector database.
        Args:
            document_ids (list): List of document IDs to delete.
        """
        pass

    @abstractmethod
    def get_retriever(self, k=3):
        """
        Return a retriever for similarity search.
        Args:
            k (int): Number of top results to retrieve.
        """
        pass