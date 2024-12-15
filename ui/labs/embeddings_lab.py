import os
from ui.labs.app_mode import AppMode
from config import Config
from embeddings.openai_embedding_model import OpenAIEmbeddingModel
from embeddings.huggingface_embedding_model import HuggingFaceEmbeddingModel
from vector_databases.chroma_vector_database import ChromaVectorDatabase
from vector_databases.pinecone_vector_database import PineconeVectorDatabase
from file_handler.file_change_detector import FileChangeDetector
import streamlit as st

class EmbeddingsLab(AppMode):

    @staticmethod
    def render():
        # Sidebar configuration
        st.sidebar.header("Embeddings Configuration")
        
        # Model Selection
        embedding_provider = st.sidebar.selectbox(
            "Select Embedding Provider",
            ["HuggingFace", "OpenAI"]
        )

        # Model-specific configuration
        embedding_model = None
        if embedding_provider == "HuggingFace":
            model_name = st.sidebar.selectbox(
                "Select HuggingFace Model",
                list(HuggingFaceEmbeddingModel.HUGGINGFACE_MODELS.keys()),
                format_func=lambda x: f"{x} ({HuggingFaceEmbeddingModel.HUGGINGFACE_MODELS[x]} dimensions)"
            )
            st.sidebar.info(f"Model dimensions: {HuggingFaceEmbeddingModel.HUGGINGFACE_MODELS[model_name]}")
            embedding_model = HuggingFaceEmbeddingModel(model_name=model_name).load_model()
        else:  # OpenAI
            #api_key = st.sidebar.text_input("OpenAI API Key", type="password")
            api_key = Config.OPENAI_API_KEY
            if api_key:
                embedding_model = OpenAIEmbeddingModel(api_key=api_key).load_model()

        # Vector Database Selection
        vector_db_type = st.sidebar.selectbox(
            "Select Vector Database",
            ["Local (ChromaDB)", "Pinecone"]
        )

        # Only show document processing for local DB
        if vector_db_type == "Local (ChromaDB)":
            st.subheader("Local Document Processing")
            
            if embedding_model:
                if st.button("Process Documents"):
                    with st.spinner("Processing documents..."):

                        # Initialize vector DB
                        vector_db = ChromaVectorDatabase(
                            persist_directory=Config.CHROMA_PERSIST_DIR_PATH,
                            embedding_model=embedding_model
                        )
                        EmbeddingsLab._process_documents(vector_db)
            else:
                st.warning("Please configure embedding model first.")

        else:  # Pinecone
            st.subheader("Pinecone Configuration")
            #api_key = st.text_input("Pinecone API Key", type="password")
            api_key = Config.PINECONE_API_KEY
            #environment = st.text_input("Pinecone Environment")
            #index_name = st.text_input("Index Name")
            index_name = Config.PINECONE_INDEX_NAME
            
            # Picone implementation TBD
            if embedding_model:
                if st.button("Process Documents"):
                    with st.spinner("Processing documents..."):
                 
                        # Initialize vector DB
                        vector_db = PineconeVectorDatabase(
                            api_key=api_key,
                            index_name=index_name,
                            embedding_model=embedding_model
                        )

                        EmbeddingsLab._process_documents(vector_db)
            else:
                st.warning("Please configure embedding model first.")

        # visualization or analysis tools TBD
        st.subheader("Embedding Analysis Tools")
        # tools for visualizing and analyzing embeddings
        # For example:
        # - Dimensionality reduction plots
        # - Similarity search demos
        # - Embedding statistics

    @staticmethod
    def _process_documents(vector_db):
        """
        Process documents for embedding generation and storage
        
        Args:
            vector_db: Initialized vector database instance
        """
        try:
            # Initialize file detector
            file_detector = FileChangeDetector(
                data_dir=Config.KNOWLEDGE_ARTICLES_DIR_PATH,
                metadata_file=Config.METADATA_FILE_PATH
            )

            # Process new/updated files
            new_or_updated_files = file_detector.detect_changes()
            valid_documents = file_detector.filter_empty_documents(new_or_updated_files)

            if not valid_documents:
                st.info("No valid documents to process.")
                vector_db.load_or_initialize(documents=[])
            else:
                st.info(f"Found {len(valid_documents)} documents to process.")
                vector_db.load_or_initialize(documents=valid_documents)
                vector_db.add_documents(valid_documents)
                st.success(f"Successfully processed {len(valid_documents)} documents.")

            # Handle deleted files
            deleted_files = file_detector.detect_deleted_files()
            if deleted_files:
                st.info(f"Removing embeddings for {len(deleted_files)} deleted files...")
                vector_db.delete_documents(deleted_files)
                st.success("Deleted file embeddings removed from the vector database.")

            # Save metadata
            file_detector.save_metadata()
            st.success("Incremental update completed successfully!")
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            raise