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

# rag_pipeline.py
from typing import Dict, Any, List
import streamlit as st
from ui.labs.app_mode import AppMode
from ui.labs.lab_utils import LabUtils
from rag.rag_processor import RAGProcessor
from config import Config, ConfigException

class RAGPipeline(AppMode):
    """
    RAG Pipeline
    """
    llm_provider: str = None
    llm_model: str = None
    llm_api_key: str = None
    embedding_option: str = None
    embedding_api_key: str = None
    vector_db_option: str = None
    vector_db_api_key: str = None
    vector_db_index: str = None
    toxicity_option: str = None
    toxicity_api_key: str = None
    preprocess_options: List[str] = None
    postprocess_options: List[str] = None
    settings: Dict[str, Any] = None

    try:
        configs = Config.load_all_configs()
        TASK_SETTINGS = configs["task_settings"]
    except ConfigException as e:
        raise ConfigException(f"Error initializing advanced configurations: {str(e)}") from e
    
    @staticmethod
    def render():
        st.header("RAG Pipeline")
        # Sidebar Options
        st.sidebar.header("Configuration Options")

        # Select LLM and get appropriate API key
        RAGPipeline.llm_provider = st.sidebar.selectbox(
            "Select LLM Provider", ["OpenAI", "Anthropic"])
        
        # Conditional model selection based on provider
        if RAGPipeline.llm_provider == "OpenAI":
            RAGPipeline.llm_model = st.sidebar.selectbox(
                "Select LLM Model", ["gpt-4", "gpt-3.5-turbo"])
        else:  # Anthropic
            RAGPipeline.llm_model = st.sidebar.selectbox(
                "Select LLM Model", ["claude-3-opus-20240229", "claude-3-sonnet-20240229"])
 
        if RAGPipeline.llm_provider.startswith("OpenAI") or RAGPipeline.llm_provider.startswith("Anthropic"):
            RAGPipeline.llm_api_key = st.sidebar.text_input(f"{RAGPipeline.llm_provider} API Key", type="password")

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
                RAGPipeline.llm_provider, RAGPipeline.llm_model, 
                RAGPipeline.llm_api_key, RAGPipeline.settings,
                RAGPipeline.embedding_option, RAGPipeline.embedding_api_key,
                RAGPipeline.vector_db_option, RAGPipeline.vector_db_api_key, RAGPipeline.vector_db_index,
                RAGPipeline.toxicity_option, RAGPipeline.toxicity_api_key,
                RAGPipeline.preprocess_options, RAGPipeline.postprocess_options
            )

            st.success(f"Using LLM: {RAGPipeline.llm_provider} {RAGPipeline.llm_model}")

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

    @staticmethod
    def _render_advanced_settings() -> Dict[str, Any]:
        settings = {}
        # Add default settings
        for setting_name, setting_config in RAGPipeline.TASK_SETTINGS["default"].items():
            if setting_config["type"] in ["float", "int"]:
                settings[setting_name] = st.slider(
                    *LabUtils.get_slider_params(setting_name, setting_config))
        
        return settings
