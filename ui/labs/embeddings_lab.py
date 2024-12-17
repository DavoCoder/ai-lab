import os
from ui.labs.app_mode import AppMode
from config import Config
from embeddings.openai_embedding_model import OpenAIEmbeddingModel
from embeddings.huggingface_embedding_model import HuggingFaceEmbeddingModel
from vector_databases.chroma_vector_database import ChromaVectorDatabase
from vector_databases.pinecone_vector_database import PineconeVectorDatabase
from file_handler.file_change_detector import FileChangeDetector
import streamlit as st
from ui.labs.embeddings_visualizer import EmbeddingsVisualizer
from web_research.web_researcher import WebResearcher
from langchain.docstore.document import Document
from data_processing.document_processor import DocumentProcessor

class EmbeddingsLab(AppMode):

    @staticmethod
    def render():
        # Sidebar configuration
        st.sidebar.header("Embeddings Configuration")
        
        #Embedding Source Selection
        embedding_source = st.sidebar.selectbox(
            "Select Embedding Source",
            ["Uploaded Documents", "Web Research", "Local Knowledge Base"]
        )
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
            api_key = st.sidebar.text_input("OpenAI API Key", type="password")
            if api_key:
                embedding_model = OpenAIEmbeddingModel(api_key=api_key).load_model()

        # Vector Database Selection
        vector_db_type = st.sidebar.selectbox(
            "Select Vector Database",
            ["Local (ChromaDB)", "Pinecone"]
        )

         # Main content area based on source selection
        if embedding_source == "Uploaded Documents":
            EmbeddingsLab._handle_uploaded_documents(embedding_model, vector_db_type)
            
        elif embedding_source == "Web Research":
            EmbeddingsLab._handle_web_research(embedding_model, vector_db_type)
            
        else:  # Local Knowledge Base
            EmbeddingsLab._handle_local_knowledge_base(embedding_model, vector_db_type)
    

    @staticmethod
    def _handle_local_knowledge_base(embedding_model, vector_db_type):
        st.subheader("Local Knowledge Base")

        if vector_db_type == "Pinecone":
            api_key = st.text_input("Pinecone API Key", type="password")
            index_name = st.text_input("Index Name")
        else: # Local (ChromaDB)
            api_key = None
            index_name = None

        tabs = st.tabs(["Document Processing", "Visualization"])

        if os.path.exists(Config.CHROMA_PERSIST_DIR_PATH):
            if embedding_model:
                vector_db = EmbeddingsLab._initialize_vector_db(
                    vector_db_type, 
                    embedding_model,
                    api_key=api_key,
                    index_name=index_name   
                )

                with tabs[0]:
                    if st.button("Process Documents"):
                        with st.spinner("Processing documents..."):
                            try:
                                EmbeddingsLab._process_local_documents(vector_db)
                            except Exception as e:
                                st.error(f"Error processing documents: {str(e)}")

                with tabs[1]:
                    if vector_db_type == "Local (ChromaDB)":
                        EmbeddingsVisualizer.visualize_embeddings(vector_db)
            else:
                st.warning("Please configure embedding model first.")
        else:
            st.warning("No vector database found. Please process some documents first.")    

    @staticmethod
    def _handle_uploaded_documents(embedding_model, vector_db_type):
        st.subheader("Document Upload")
        print("vector_db_type: ", vector_db_type)
        print("embedding_model: ", embedding_model)

        if vector_db_type == "Pinecone":
            api_key = st.text_input("Pinecone API Key", type="password")
            index_name = st.text_input("Index Name")
        else: # Local (ChromaDB)
            api_key = None
            index_name = None
        
        uploaded_files = st.file_uploader(
            "Upload Documents", 
            accept_multiple_files=True,
            type=["txt", "pdf", "docx"]
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Documents"):
                with st.spinner("Processing uploaded documents..."):
                    try:
                        vector_db = EmbeddingsLab._initialize_vector_db(
                            vector_db_type, 
                            embedding_model,
                            api_key=api_key,
                            index_name=index_name   
                        )
                        
                        print("Vector DB initialized")
                        # Process documents
                        documents = []
                        for file in uploaded_files:
                            content = DocumentProcessor.read_file_content(file)
                            documents.append(
                               Document(
                                    page_content=content,
                                    metadata={
                                        "source": file.name,
                                        "type": "uploaded_document"
                                    }
                                )
                            )
                        
                        print("Documents uploaded")
                        print("Vector DB Load or Initialize")
                        # Add to vector DB
                        vector_db.load_or_initialize(documents=[])

                        print("Vector DB Add Documents")
                        vector_db.add_documents(documents)
                        st.success(f"Successfully processed {len(documents)} documents")
                        
                        
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
    
    @staticmethod
    def _handle_web_research(embedding_model, vector_db_type):
        st.subheader("Web Research")
        
        # Web research configuration
        search_query = st.text_area("Research Query")
        urls = st.text_area("Enter URLs (one per line)")
        
        if search_query and urls:
            if st.button("Process Web Content"):
                with st.spinner("Processing web content..."):
                    try:
                        # Initialize vector DB
                        vector_db = EmbeddingsLab._initialize_vector_db(
                            vector_db_type, 
                            embedding_model
                        )
                        
                        # Process web content
                        web_researcher = WebResearcher()
                        results = web_researcher.search(
                            query=search_query,
                            urls=urls.strip().split('\n'),
                            model_provider="OpenAI",  # Could be made configurable
                            model_id="gpt-3.5-turbo"  # Could be made configurable
                        )
                        
                        # Prepare documents
                        documents = []
                        for url, content in results.items():
                            documents.append({
                                "content": content,
                                "metadata": {
                                    "source": url,
                                    "type": "web_research",
                                    "query": search_query
                                }
                            })
                        
                        # Add to vector DB
                        vector_db.load_or_initialize(documents=[])
                        vector_db.add_documents(documents)
                        st.success(f"Successfully processed {len(documents)} web sources")
                        
                    except Exception as e:
                        st.error(f"Error processing web content: {str(e)}")

        
    @staticmethod
    def _initialize_vector_db(vector_db_type: str, embedding_model, api_key: str, index_name: str):
        """Initialize vector database with appropriate configuration."""
        if vector_db_type == "Local (ChromaDB)":
            return ChromaVectorDatabase(
                persist_directory=Config.CHROMA_PERSIST_DIR_PATH,
                embedding_model=embedding_model
            )
        else:  # Pinecone
            return PineconeVectorDatabase(
                api_key=api_key,
                index_name=index_name,
                embedding_model=embedding_model
            )
    
    @staticmethod
    def _process_local_documents(vector_db):
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
