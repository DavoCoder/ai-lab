import pinecone
from langchain_community.vectorstores import Pinecone as LangChainPinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from vector_databases.vector_database_interface import VectorDatabase

class PineconeVectorDatabase(VectorDatabase):
    def __init__(self, index_name, embedding_model, api_key, environment):
        pinecone.init(api_key=api_key, environment=environment)
        self.index_name = index_name
        self.embedding_model = embedding_model

        if index_name not in pinecone.list_indexes():
            print(f"Creating Pinecone index '{index_name}'...")
            pinecone.create_index(index_name, dimension=1536)  # Adjust for embedding model

        self.index = pinecone.Index(index_name)
        self.vector_store = LangChainPinecone(self.index, self.embedding_model)

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
