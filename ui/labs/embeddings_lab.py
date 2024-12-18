import os
import streamlit as st
from ui.labs.app_mode import AppMode
from embeddings.huggingface_embedding_model import HuggingFaceEmbeddingModel
from ui.labs.embeddings_visualizer import EmbeddingsVisualizer
from embeddings.embedding_generator import EmbeddingGenerator
class EmbeddingsLab(AppMode):

    embedding_provider = None
    embedding_model_name = None
    embedding_api_key = None
    vector_db_type = None
    vector_db_api_key = None
    vector_db_index = None
    web_research_eval_provider = None
    web_research_eval_model_id = None
    web_research_eval_provider_api_key = None

    @staticmethod
    def render():
        # Sidebar configuration
        st.sidebar.header("Embeddings Configuration")
        
        #Embedding Source Selection
        embedding_source = st.sidebar.selectbox(
            "Select Embedding Source",
            ["Uploaded Documents", "Web Research", "Local Knowledge Base"]
        )

        if embedding_source == "Web Research":
            # Embedding Provider Selection
            EmbeddingsLab.web_research_eval_provider = st.sidebar.selectbox(
                "Select Web Research Provider",
                ["OpenAI"]
            )
            if EmbeddingsLab.web_research_eval_provider == "OpenAI":
                EmbeddingsLab.web_research_eval_model_id = st.sidebar.selectbox(
                    "Select Web Research Model",
                    ["gpt-4", "gpt-3.5-turbo"]
                )
                EmbeddingsLab.web_research_eval_provider_api_key = st.sidebar.text_input(
                    f"{EmbeddingsLab.web_research_eval_provider} API Key for Web Research", type="password")
            else:
                raise NotImplementedError(f"Provider {EmbeddingsLab.web_research_eval_provider} not implemented yet")


        # Embedding Provider Selection
        EmbeddingsLab.embedding_provider = st.sidebar.selectbox(
            "Select Embedding Provider",
            ["HuggingFace", "OpenAI"]
        )

        # Model-specific configuration
        if EmbeddingsLab.embedding_provider == "HuggingFace":
            EmbeddingsLab.embedding_model_name = st.sidebar.selectbox(
                "Select HuggingFace Model",
                list(HuggingFaceEmbeddingModel.HUGGINGFACE_MODELS.keys()),
                format_func=lambda x: f"{x} ({HuggingFaceEmbeddingModel.HUGGINGFACE_MODELS[x]} dimensions)"
            )
            st.sidebar.info(f"Model dimensions: {HuggingFaceEmbeddingModel.HUGGINGFACE_MODELS[EmbeddingsLab.embedding_model_name]}")
        else: 
            EmbeddingsLab.embedding_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
            if EmbeddingsLab.embedding_api_key:
                EmbeddingsLab.embedding_model_name = "OpenAI"

        

        # Vector Database Selection
        EmbeddingsLab.vector_db_type = st.sidebar.selectbox(
            "Select Vector Database",
            ["Local (ChromaDB)", "Pinecone"]
        )

        if EmbeddingsLab.vector_db_type == "Pinecone":
            EmbeddingsLab.vector_db_api_key = st.sidebar.text_input("Pinecone API Key", type="password")
            EmbeddingsLab.vector_db_index = st.sidebar.text_input("Index Name")


         # Main content area based on source selection
        if embedding_source == "Uploaded Documents":
            EmbeddingsLab._handle_uploaded_documents()
            
        elif embedding_source == "Web Research":
            EmbeddingsLab._handle_web_research()
            
        else:  # Local Knowledge Base
            EmbeddingsLab._handle_local_knowledge_base()
    

    @staticmethod
    def _handle_local_knowledge_base():
        st.subheader("Local Knowledge Base")

        tabs = st.tabs(["Document Processing", "Visualization"])

        embeddingGenerator = EmbeddingGenerator(
            embedding_option=EmbeddingsLab.embedding_provider,
            embedding_model_name=EmbeddingsLab.embedding_model_name,
            embedding_api_key=EmbeddingsLab.embedding_api_key,
            vector_db_option=EmbeddingsLab.vector_db_type,
            vector_db_api_key=EmbeddingsLab.vector_db_api_key,
            vector_db_index=EmbeddingsLab.vector_db_index,
            web_research_provider=None,
            web_research_model_id=None,
            web_research_provider_api_key=None
        )
        with tabs[0]:
            if st.button("Process Documents"):
                with st.spinner("Processing documents..."):
                    try:
                        results = embeddingGenerator.process_local_knowledge_base()
                        st.success(results)
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")

        with tabs[1]:
            if embeddingGenerator.vector_db_option == "Local (ChromaDB)":
                EmbeddingsVisualizer.visualize_embeddings(embeddingGenerator.vector_db)


    @staticmethod
    def _handle_uploaded_documents():
        st.subheader("Document Upload")
            
        uploaded_files = st.file_uploader(
            "Upload Documents", 
            accept_multiple_files=True,
            type=["txt", "pdf", "docx"]
        )
        
        if uploaded_files:
            if st.button("Process Uploaded Documents"):
                with st.spinner("Processing uploaded documents..."):
                    try:
                        embeddingGenerator = EmbeddingGenerator(
                            embedding_option=EmbeddingsLab.embedding_provider,
                            embedding_model_name=EmbeddingsLab.embedding_model_name,
                            embedding_api_key=EmbeddingsLab.embedding_api_key,
                            vector_db_option=EmbeddingsLab.vector_db_type,
                            vector_db_api_key=EmbeddingsLab.vector_db_api_key,
                            vector_db_index=EmbeddingsLab.vector_db_index,
                            web_research_provider=None,
                            web_research_model_id=None,
                            web_research_provider_api_key=None
                        )
                        results = embeddingGenerator.process_uploaded_documents(uploaded_files)
                        st.success(results)

                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
    
    @staticmethod
    def _handle_web_research():
        st.subheader("Web Research")
        
        # Web research configuration
        search_query = st.text_area("Research Query")
        urls = st.text_area("Enter URLs (one per line)")
        
        if search_query and urls:
            if st.button("Process Web Content"):
                with st.spinner("Processing web content..."):
                    try:
                        embeddingGenerator = EmbeddingGenerator(
                            embedding_option=EmbeddingsLab.embedding_provider,
                            embedding_model_name=EmbeddingsLab.embedding_model_name,
                            embedding_api_key=EmbeddingsLab.embedding_api_key,
                            vector_db_option=EmbeddingsLab.vector_db_type,
                            vector_db_api_key=EmbeddingsLab.vector_db_api_key,
                            vector_db_index=EmbeddingsLab.vector_db_index,
                            web_research_provider=EmbeddingsLab.web_research_eval_provider,
                            web_research_model_id=EmbeddingsLab.web_research_eval_model_id,
                            web_research_provider_api_key=EmbeddingsLab.web_research_eval_provider_api_key
                        )
                        results = embeddingGenerator.process_web_research(
                            search_query=search_query, 
                            urls=urls
                        )
                        st.success(results)
                        
                    except Exception as e:
                        st.error(f"Error processing web content: {str(e)}")

        