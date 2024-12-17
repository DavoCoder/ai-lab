from ui.labs.app_mode import AppMode
import streamlit as st
from embeddings.openai_embedding_model import OpenAIEmbeddingModel
from embeddings.huggingface_embedding_model import HuggingFaceEmbeddingModel
from vector_databases.chroma_vector_database import ChromaVectorDatabase
from vector_databases.pinecone_vector_database import PineconeVectorDatabase
from toxicity_detection.huggingface_toxicity_detector import HuggingFaceToxicityDetector
from toxicity_detection.openai_toxicity_detector import OpenAIToxicityDetector
from query_pre_processing.spellcheck_query_processor import SpellCheckQueryProcessor
from query_pre_processing.query_rewriter_processor import QueryRewriterProcessor
from response_post_processing.hallucination_filter import HallucinationFilter
from response_post_processing.summarization_post_processor import SummarizationPostProcessor
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from config import Config
from typing import Dict, Any

class RAGPipeline(AppMode):

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
        llm_option = st.sidebar.selectbox("Select LLM", ["OpenAI GPT-4", "OpenAI GPT-3.5", "Anthropic Claude-3 Opus", "Anthropic Claude-3 Sonnet"])
 
        llm_api_key = None
        if llm_option in ["OpenAI", "Anthropic"]:
            llm_api_key = st.sidebar.text_input(f"{llm_option} API Key", type="password")

        # Select Embedding Model
        embedding_option = st.sidebar.selectbox("Select Embedding Model", ["OpenAI", "HuggingFace"])
        embedding_api_key = None
        if embedding_option == "OpenAI":
            embedding_api_key = st.sidebar.text_input("OpenAI API Key for Embeddings", type="password", key="embed_openai_key")

        # Select Vector Database
        vector_db_option = st.sidebar.selectbox("Select Vector Database", ["ChromaDB", "Pinecone"])
        vector_db_api_key = None
        vector_db_index = None
        if vector_db_option == "Pinecone":
            vector_db_api_key = st.sidebar.text_input("Pinecone API Key", type="password")
            vector_db_index = st.sidebar.text_input("Pinecone Index Name")

        # Select Toxicity Detection
        toxicity_option = st.sidebar.selectbox("Toxicity Detection", ["None", "OpenAI", "HuggingFace"])
        toxicity_api_key = None
        if toxicity_option == "OpenAI":
            toxicity_api_key = st.sidebar.text_input("OpenAI API Key for Toxicity Detection", type="password", key="tox_openai_key")

        # Select Pre-Processing Options (Multi-Select)
        preprocess_options = st.sidebar.multiselect(
            "Select Query Pre-Processing Steps",
            ["Spell Check", "Query Rewriter"]
        )

        # Select Post-Processing Options (Multi-Select)
        postprocess_options = st.sidebar.multiselect(
            "Select Response Post-Processing Steps",
            ["Hallucination Filter", "Summarization"]
        )

        # Advanced Settings
        with st.sidebar.expander("Advanced Settings"):
            settings = RAGPipeline._render_advanced_settings()

        # Input Query
        user_query = st.text_input("Enter your query:")

        # Submit Button
        if st.button("Submit Query"):

            # Step 1: Initialize Embedding Model
            if embedding_option == "OpenAI":
                embedding_model = OpenAIEmbeddingModel(api_key=embedding_api_key).load_model()
            elif embedding_option == "HuggingFace":
                embedding_model = HuggingFaceEmbeddingModel(model_name="all-MiniLM-L6-v2").load_model()
            st.success(f"Using Embedding Model: {embedding_option}")

            # Step 2: Initialize Vector Database
            if vector_db_option == "ChromaDB":
                vector_db = ChromaVectorDatabase(persist_directory=Config.CHROMA_PERSIST_DIR_PATH, embedding_model=embedding_model)
            elif vector_db_option == "Pinecone":
                vector_db = PineconeVectorDatabase(index_name=vector_db_index, embedding_model=embedding_model,
                                                api_key=vector_db_api_key)
            st.success(f"Using Vector Database: {vector_db_option}")

            # Step 3: Initialize Toxicity Detector
            toxicity_detector = None
            if toxicity_option == "OpenAI":
                toxicity_detector = OpenAIToxicityDetector(api_key=toxicity_api_key)
            elif toxicity_option == "HuggingFace":
                toxicity_detector = HuggingFaceToxicityDetector()
            st.success(f"Toxicity Detection: {toxicity_option}")

            # Step 4: Query Pre-Processing
            st.info("Applying Query Pre-Processing...")
            for option in preprocess_options:
                if option == "Spell Check":
                    query_processor = SpellCheckQueryProcessor()
                    user_query = query_processor.process(user_query)
                elif option == "Query Rewriter":
                    query_processor = QueryRewriterProcessor()
                    user_query = query_processor.process(user_query)
            st.success(f"Pre-processed Query: {user_query}")

            # Step 5: Check for Toxicity
            if toxicity_detector and toxicity_detector.detect_toxicity(user_query):
                st.error("Query contains toxic content. Aborting.")
                st.stop()

            # Step 6: Retrieval and Response Generation
            vector_db.load_or_initialize(documents=[])
            retriever = vector_db.get_retriever(k=3)

            # Use the function to get the LLM
            llm = RAGPipeline._get_llm(llm_option, llm_api_key, settings)

            qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

            # Retrieve documents and generate response
            with st.spinner("Retrieving documents and generating response..."):
                response = qa_chain.invoke(user_query)['result']

            # Step 7: Response Post-Processing
            st.info("Applying Response Post-Processing...")
            for option in postprocess_options:
                if option == "Hallucination Filter":
                    post_processor = HallucinationFilter()
                    response = post_processor.process(response, retriever.get_relevant_documents(user_query))
                elif option == "Summarization":
                    post_processor = SummarizationPostProcessor()
                    response = post_processor.process(response, None)
            st.success("Post-Processing Complete.")

            # Step 8: Display Final Response
            st.subheader("Response:")
            st.write(response)
    
    #TODO: Externalize this to an LLM init class in the backend
    def _get_llm(llm_option, api_key, settings):
        if "Anthropic Claude" in llm_option:
            return ChatAnthropic(
                model="claude-3-opus-20240229" if "Opus" in llm_option else "claude-3-sonnet-20240229",
                anthropic_api_key=api_key,
                temperature=settings.get("temperature", 0.7),
                max_tokens=settings.get("max_tokens", 500)
            )
        else:
            return ChatOpenAI(
                api_key=api_key,
                model="gpt-4" if "GPT-4" in llm_option else "gpt-3.5-turbo",
                temperature=settings.get("temperature", 0.7),
                max_tokens=settings.get("max_tokens", 500)
            )
        
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