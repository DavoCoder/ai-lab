from ui.labs.app_mode import AppMode
import streamlit as st
from config import Config
from typing import Dict, Any
from rag.rag_processor import RAGProcessor

class RAGPipeline(AppMode):

    llm_option = None
    llm_api_key = None
    embedding_option = None
    embedding_api_key = None
    vector_db_option = None
    vector_db_api_key = None
    vector_db_index = None
    toxicity_option = None
    toxicity_api_key = None
    preprocess_options = None
    postprocess_options = None
    settings = None

    try:
        configs = Config.load_all_configs()
        TASK_SETTINGS = configs["task_settings"]
    except Exception as e:
        raise Exception(f"Error initializing advanced configurations: {str(e)}")
    
    @staticmethod
    def render():
        st.header("RAG Pipeline")
        # Sidebar Options
        st.sidebar.header("Configuration Options")

        # Select LLM and get appropriate API key
        RAGPipeline.llm_option = st.sidebar.selectbox("Select LLM", ["OpenAI GPT-4", "OpenAI GPT-3.5", 
                                                                     "Anthropic Claude-3 Opus", "Anthropic Claude-3 Sonnet"])
 
        if "OpenAI" in RAGPipeline.llm_option or "Anthropic" in RAGPipeline.llm_option:
            RAGPipeline.llm_api_key = st.sidebar.text_input(f"{RAGPipeline.llm_option} API Key", type="password")

        # Select Embedding Model
        RAGPipeline.embedding_option = st.sidebar.selectbox("Select Embedding Model", ["OpenAI", "HuggingFace"])
        if RAGPipeline.embedding_option == "OpenAI":
            RAGPipeline.embedding_api_key = st.sidebar.text_input("OpenAI API Key for Embeddings", 
                                                                  type="password", key="embed_openai_key")
        # Select Vector Database
        RAGPipeline.vector_db_option = st.sidebar.selectbox("Select Vector Database", ["ChromaDB", "Pinecone"])
        if RAGPipeline.vector_db_option == "Pinecone":
            RAGPipeline.vector_db_api_key = st.sidebar.text_input("Pinecone API Key", type="password")
            RAGPipeline.vector_db_index = st.sidebar.text_input("Pinecone Index Name")

        # Select Toxicity Detection
        RAGPipeline.toxicity_option = st.sidebar.selectbox("Toxicity Detection", ["None", "OpenAI", "HuggingFace"])
        if RAGPipeline.toxicity_option == "OpenAI":
            RAGPipeline.toxicity_api_key = st.sidebar.text_input("OpenAI API Key for Toxicity Detection", 
                                                                 type="password", key="tox_openai_key")

        # Select Pre-Processing Options (Multi-Select)
        RAGPipeline.preprocess_options = st.sidebar.multiselect(
            "Select Query Pre-Processing Steps",
            ["Spell Check", "Query Rewriter"]
        )

        # Select Post-Processing Options (Multi-Select)
        RAGPipeline.postprocess_options = st.sidebar.multiselect(
            "Select Response Post-Processing Steps",
            ["Hallucination Filter", "Summarization"]
        )

        # Advanced Settings
        with st.sidebar.expander("Advanced Settings", expanded=True):
            RAGPipeline.settings = RAGPipeline._render_advanced_settings()

        # Input Query
        user_query = st.text_input("Enter your query:")

        # Submit Button
        if st.button("Submit Query"):

            ragProcessor = RAGProcessor(   
                RAGPipeline.llm_option, RAGPipeline.llm_api_key, RAGPipeline.settings,
                RAGPipeline.embedding_option, RAGPipeline.embedding_api_key,
                RAGPipeline.vector_db_option, RAGPipeline.vector_db_api_key, RAGPipeline.vector_db_index,
                RAGPipeline.toxicity_option, RAGPipeline.toxicity_api_key,
                RAGPipeline.preprocess_options, RAGPipeline.postprocess_options
            )
             # Step 1: Initialize Embeddings        
            ragProcessor.initialize_embeddings()
            st.success(f"Using Embedding Model: {RAGPipeline.embedding_option}")

            # Step 2: Initialize Vector Database
            ragProcessor.initialize_vector_db()
            st.success(f"Using Vector Database: {RAGPipeline.vector_db_option}")

            #Step 3: Initialize Toxicity Detection
            ragProcessor.initialize_toxicity_detector()
            st.success(f"Using Toxicity Detection: {RAGPipeline.toxicity_option}")

            st.success(f"Using Pre-Processing options: {RAGPipeline.preprocess_options}")
            st.success(f"Using Post-Processing options: {RAGPipeline.postprocess_options}")

            # Step 4: Apply Pre-Processing
            user_query = ragProcessor.apply_pre_processing(user_query)
            st.success("Pre-Processing completed")
            
            # Step 5: Check for Toxicity
            if ragProcessor.apply_toxicity_detection(user_query):
                st.error("Query contains toxic content. Aborting.")
                st.stop()

            # Step 6: Retrieve documents and generate response
            with st.spinner("Retrieving documents and generating response..."):
                response = ragProcessor.execute_qa_chain(user_query)

            #Step 7: Apply Response Post-Processing
            st.info("Applying Response Post-Processing...")
            response = ragProcessor.apply_post_processing(response, user_query)
            st.success("Post-Processing completed")

            # Step 8: Display Final Response
            st.subheader("Response:")
            st.write(response)

        
    def _render_advanced_settings() -> Dict[str, Any]:
        settings = {}
        # Add default settings
        for setting_name, setting_config in RAGPipeline.TASK_SETTINGS["default"].items():
            if setting_config["type"] in ["float", "int"]:
                settings[setting_name] = st.slider(
                    setting_name.replace("_", " ").title(),
                    setting_config["min"],
                    setting_config["max"],
                    setting_config["default"]
                )
        
        
        return settings