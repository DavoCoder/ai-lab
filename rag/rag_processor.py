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

# rag_processor.py
import logging
from typing import Dict, Any
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
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
from config import Config


class RAGProcessor:
    """
    Processes RAG (Retrieval Augmented Generation) operations using LangChain.
    Handles model initialization, vector stores, and QA chain creation.
    """

    def __init__(
        self,
        llm_option: str,
        llm_api_key: str,
        llm_settings: Dict[str, Any],
        embedding_option: str,
        embedding_api_key: str,
        vector_db_option: str,
        vector_db_api_key: str,
        vector_db_index: str,
        toxicity_option: str,
        toxicity_api_key: str,
        pre_process_options: list,
        post_process_options: list,
    ):
        self.llm_option = llm_option
        self.llm_api_key = llm_api_key
        self.llm_settings = llm_settings
        self.embedding_option = embedding_option
        self.embedding_api_key = embedding_api_key
        self.vector_db_option = vector_db_option
        self.vector_db_api_key = vector_db_api_key
        self.vector_db_index = vector_db_index
        self.toxicity_option = toxicity_option
        self.toxicity_api_key = toxicity_api_key
        self.pre_process_options = pre_process_options
        self.post_process_options = post_process_options

        # Initialize as None, will be set up when needed
        self.embedding_model = None
        self.vector_db = None
        self.toxicity_detector = None
        self.llm = None
        self.retriever = None
        self.qa_chain = None

        logging.info("Initializing RAGProcessor with LLM: %s, Embedding: %s, Vector DB: %s", 
                     llm_option, embedding_option, vector_db_option)

    def initialize_and_execute_all(self, user_query: str) -> str:
        logging.info("Processing query: %s", user_query)
        self.initialize_embeddings()
        self.initialize_vector_db()
        self.initialize_toxicity_detector()
       
        processed_user_query = self.apply_pre_processing(user_query)
        logging.info("Processed query: %s", processed_user_query)
          
        if self.apply_toxicity_detection(processed_user_query):
            logging.warning("Toxic content detected in processed query: %s", processed_user_query)
            return "Query contains toxic content. Aborting."
          
        response = self.execute_qa_chain(processed_user_query)
        logging.info("Raw response from QA chain: %s", response)
        
        response = self.apply_post_processing(response, processed_user_query)
        logging.info("Final processed response: %s", response)

        return response
    
    def _get_llm(self):
        if "Anthropic Claude" in self.llm_option:
            return ChatAnthropic(
                model="claude-3-opus-20240229" if "Opus" in self.llm_option else "claude-3-sonnet-20240229",
                anthropic_api_key=self.llm_api_key,
                temperature=self.llm_settings.get("temperature", 0.7),
                max_tokens=self.llm_settings.get("max_tokens", 500)
            )
        if "OpenAI GPT" in self.llm_option:
            return ChatOpenAI(
                api_key=self.llm_api_key,
                model="gpt-4" if "GPT-4" in self.llm_option else "gpt-3.5-turbo",
                temperature=self.llm_settings.get("temperature", 0.7),
                max_tokens=self.llm_settings.get("max_tokens", 500)
            )
       
        raise RAGProcessorException(f"Unsupported model: {self.llm_option}")

    def initialize_embeddings(self) -> None:
        """Initialize the embedding model."""
        
        logging.info("Initializing embeddings with option: %s", self.embedding_option)
        
        if self.embedding_option == "OpenAI":
            self.embedding_model = OpenAIEmbeddingModel(api_key=self.embedding_api_key).load_model()
        elif self.embedding_option == "HuggingFace":
            self.embedding_model = HuggingFaceEmbeddingModel(model_name="all-MiniLM-L6-v2").load_model()

    def initialize_vector_db(self) -> None:
        """Create a new vector store or load existing one."""
        
        logging.info("Initializing vector database: %s", self.vector_db_option)
        
        if self.embedding_model is None:
            self.initialize_embeddings()
        
        if self.vector_db_option == "ChromaDB":
            self.vector_db = ChromaVectorDatabase(persist_directory=Config.CHROMA_PERSIST_DIR_PATH, 
                                                  embedding_model=self.embedding_model)
        elif self.vector_db_option == "Pinecone":
            self.vector_db = PineconeVectorDatabase(index_name=self.vector_db_index, 
                                                    embedding_model=self.embedding_model,
                                                    api_key=self.vector_db_api_key)
        else:
            raise RAGProcessorException(f"Unsupported vector database option: {self.vector_db_option}")
        
    def initialize_toxicity_detector(self):
        logging.info("Initializing toxicity detector: %s", self.toxicity_option)
        if self.toxicity_option == "OpenAI":
            self.toxicity_detector = OpenAIToxicityDetector(api_key=self.toxicity_api_key)
        elif self.toxicity_option == "HuggingFace":
            self.toxicity_detector = HuggingFaceToxicityDetector()

    def apply_toxicity_detection(self, user_query: str) -> str:
        if self.toxicity_detector:
            return self.toxicity_detector.detect_toxicity(user_query)
        return False

    def apply_pre_processing(self, user_query: str) -> str:
        logging.info("Applying pre-processing steps: %s", self.pre_process_options)
        for option in self.pre_process_options:
            if option == "Spell Check":
                query_processor = SpellCheckQueryProcessor()
                user_query = query_processor.process(user_query)
            elif option == "Query Rewriter":
                query_processor = QueryRewriterProcessor()
                user_query = query_processor.process(user_query)
        return user_query
    
    def execute_qa_chain(self, user_query: str) -> str:
        logging.info("Executing QA chain")
        self.vector_db.load_or_initialize(documents=[])
        self.retriever = self.vector_db.get_retriever(k=3)
        self.llm = self._get_llm()
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", retriever=self.retriever)
       
        return self.qa_chain.invoke(user_query)["result"]

    def apply_post_processing(self, response: str, user_query: str) -> str:
        logging.info("Applying post-processing steps: %s", self.post_process_options)
        for option in self.post_process_options:
            if option == "Hallucination Filter":
                post_processor = HallucinationFilter()
                response = post_processor.process(response, self.retriever.get_relevant_documents(user_query))
            elif option == "Summarization":
                post_processor = SummarizationPostProcessor()
                response = post_processor.process(response, None)
        return response

class RAGProcessorException(Exception):
    """Base exception for RAG processor related errors"""
    def __init__(self, message="RAG processor error occurred"):
        self.message = message
        super().__init__(self.message)
