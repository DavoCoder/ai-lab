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

# pinecone_vector_database.py
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_databases.vector_database_interface import VectorDatabase

class PineconeVectorDatabase(VectorDatabase):
    def __init__(self, api_key, index_name, embedding_model):
        self.pc = Pinecone(api_key=api_key )
        self.index_name = index_name
        self.embedding_model = embedding_model

        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating Pinecone index '{index_name}'...")
            # TODO: get dimensions from user input
            self.pc.create_index(
                name=self.index_name, 
                dimension=1536, 
                metric='euclidean',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-west-2'
                )
            )

        self.index = self.pc.Index(index_name)
        self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embedding_model)

    def load_or_initialize(self, documents=None):
        print("Pinecone index ready. No initialization needed.")

    def add_documents(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        texts = text_splitter.split_documents(documents)
        self.vector_store.add_documents(texts)
        print("Documents added to Pinecone.")

    def delete_documents(self, document_ids):
        self.index.delete(ids=document_ids)
        print("Documents deleted from Pinecone.")

    def get_retriever(self, k=3):
        return self.vector_store.as_retriever(search_kwargs={"k": k})
