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

# chroma_vector_database.py
import os
from typing import List, Dict, Any
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_databases.vector_database_interface import VectorDatabase

class ChromaVectorDatabase(VectorDatabase):
    def __init__(self, persist_directory, embedding_model):
        """
        Initialize the vector database.

        Args:
            persist_directory (str): Directory to persist the database.
            embedding_model: The embedding model instance to use.
        """
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.vector_db = None

    def load_or_initialize(self, documents=None):
        """
        Load an existing vector database or initialize a new one.

        Args:
            documents: Documents to add to the database if initializing.
        """
        if os.path.exists(self.persist_directory):
            print("Loading existing VectorDB...")
            self.vector_db = Chroma(persist_directory=self.persist_directory, 
                                   embedding_function=self.embedding_model)
        else:
            print("Creating new VectorDB...")
            if not documents:
                raise ValueError("Documents must be provided when initializing a new VectorDB.")
            self.vector_db = Chroma.from_documents(documents, 
                                                   self.embedding_model, 
                                                   persist_directory=self.persist_directory)

    def add_documents(self, documents):
        """
        Add new documents to the vector database.
        """
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)

        # Check for empty text chunks
        if not texts:
            raise ValueError("No valid text chunks found to add to the VectorDB.")

        print(f"Number of text chunks created: {len(texts)}")

        self.vector_db.add_documents(documents)
        # For the latest version of Chroma, persistence is handled by the collection
        if hasattr(self.vector_db, 'collection'):
            self.vector_db.collection.persist()

        print("New documents added and persisted.")

    def delete_documents(self, document_ids):
        self.vector_db.delete(document_ids) 
        print("Documents deleted from ChromaDB.")

    def get_retriever(self, k=3):
        """
        Return the retriever interface for similarity search.

        Args:
            k (int): Number of top documents to retrieve.
        """
        if not self.vector_db:
            raise ValueError("VectorDB is not initialized. Call load_or_initialize() first.")
        return self.vector_db.as_retriever(search_kwargs={"k": k})

    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Perform similarity search using ChromaDB."""
        if not self.vector_db:
            raise ValueError("Database not initialized. Call load_or_initialize first.")
        
        # Perform search
        results = self.vector_db.similarity_search(
            query=query,
            n_results=k
        )
        
        # Format results
        formatted_results = []
        for idx in range(len(results['documents'][0])):
            formatted_results.append({
                "text": results['documents'][0][idx],
                "metadata": results['metadatas'][0][idx] if results['metadatas'] else {}
            })
            
        return formatted_results
