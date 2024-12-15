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

class RAGPipeline(AppMode):
    @staticmethod
    def render():
        st.header("RAG Pipeline")
        # Sidebar Options
        st.sidebar.header("Configuration Options")

        # Select LLM
        llm_option = st.sidebar.selectbox("Select LLM", ["OpenAI GPT-4", "OpenAI GPT-3.5", "Anthropic Claude-3 Opus", "Anthropic Claude-3 Sonnet"])

        # Select Embedding Model
        embedding_option = st.sidebar.selectbox("Select Embedding Model", ["OpenAI", "HuggingFace"])

        # Select Vector Database
        vector_db_option = st.sidebar.selectbox("Select Vector Database", ["ChromaDB", "Pinecone"])

        # Select Pre-Processing Options (Multi-Select)
        preprocess_options = st.sidebar.multiselect(
            "Select Query Pre-Processing Steps",
            ["Spell Check", "Query Rewriter"]
        )

        # Select Toxicity Detection
        toxicity_option = st.sidebar.selectbox("Toxicity Detection", ["None", "OpenAI", "HuggingFace"])

        # Select Post-Processing Options (Multi-Select)
        postprocess_options = st.sidebar.multiselect(
            "Select Response Post-Processing Steps",
            ["Hallucination Filter", "Summarization"]
        )

        # Input Query
        user_query = st.text_input("Enter your query:")

        # Submit Button
        if st.button("Submit Query"):

            # Step 1: Initialize Embedding Model
            if embedding_option == "OpenAI":
                embedding_model = OpenAIEmbeddingModel(api_key=Config.OPENAI_API_KEY).load_model()
            elif embedding_option == "HuggingFace":
                embedding_model = HuggingFaceEmbeddingModel(model_name="all-MiniLM-L6-v2").load_model()
            st.success(f"Using Embedding Model: {embedding_option}")

            # Step 2: Initialize Vector Database
            if vector_db_option == "ChromaDB":
                vector_db = ChromaVectorDatabase(persist_directory=Config.CHROMA_PERSIST_DIR_PATH, embedding_model=embedding_model)
            elif vector_db_option == "Pinecone":
                vector_db = PineconeVectorDatabase(index_name="rag-index", embedding_model=embedding_model,
                                                api_key="your-pinecone-api-key", environment="your-pinecone-env")
            st.success(f"Using Vector Database: {vector_db_option}")

            # Step 3: Initialize Toxicity Detector
            toxicity_detector = None
            if toxicity_option == "OpenAI":
                toxicity_detector = OpenAIToxicityDetector(api_key=Config.OPENAI_API_KEY)
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

            # Update the LLM initialization
            def get_llm(llm_option):
                if llm_option == "Anthropic Claude":
                    return ChatAnthropic(
                        model="claude-3-opus-20240229" if llm_option == "Anthropic Claude-3 Opus" else "claude-3-sonnet-20240229",
                        anthropic_api_key=Config.ANTHROPIC_API_KEY,
                        temperature=0
                    )
                else:
                    return ChatOpenAI(model="gpt-4" if llm_option == "OpenAI GPT-4" else "gpt-3.5-turbo", temperature=0)

            # Use the function to get the LLM
            llm = get_llm(llm_option)

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