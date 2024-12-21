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

# embedding_generator.py
import logging
from typing import List, Dict, Any
from langchain.schema import Document
from data_processing.document_processor import DocumentProcessor
from web_research.web_researcher import WebResearcher
from file_handler.file_change_detector import FileChangeDetector
from vector_databases.chroma_vector_database import ChromaVectorDatabase
from vector_databases.pinecone_vector_database import PineconeVectorDatabase
from embeddings.huggingface_embedding_model import HuggingFaceEmbeddingModel
from embeddings.openai_embedding_model import OpenAIEmbeddingModel
from config import Config

# Set up logger
logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """
    EmbeddingGenerator class for processing and storing documents in a vector database.
    """
    def __init__(self, embedding_option: str,
        embedding_model_name: str,
        embedding_api_key: str,
        vector_db_option: str,
        vector_db_api_key: str,
        vector_db_index: str,
        web_research_provider: str,
        web_research_model_id: str,
        web_research_provider_api_key: str):

        logger.info("Initializing EmbeddingGenerator with model: %s", embedding_model_name)

        self.embedding_option = embedding_option
        self.embedding_model_name = embedding_model_name
        self.embedding_api_key = embedding_api_key
        self.vector_db_option = vector_db_option
        self.vector_db_api_key = vector_db_api_key
        self.vector_db_index = vector_db_index
        self.web_research_provider = web_research_provider
        self.web_research_model_id = web_research_model_id
        self.web_research_provider_api_key = web_research_provider_api_key

        self.embedding_model = None
        self.vector_db = None

        try:
            self._initialize_embedding_model()
            self._initialize_vector_db()
            logger.debug("Successfully initialized embedding model and vector database")
        except Exception as e:
            logger.error("Initialization failed: %s", str(e), exc_info=True)
            raise EmbeddingGeneratorException(f"Initialization failed: {str(e)}") from e

    def _initialize_embedding_model(self):
        """Initialize the embedding model."""
        logger.debug("Initializing embedding model: %s", self.embedding_option)
    
        try:
            if self.embedding_option == "HuggingFace":
                self.embedding_model = HuggingFaceEmbeddingModel(model_name=self.embedding_model_name).load_model()
            elif self.embedding_option == "OpenAI":
                self.embedding_model = OpenAIEmbeddingModel(api_key=self.embedding_api_key).load_model()
            else:
                logger.error("Unsupported embedding model: %s", self.embedding_option)
                raise ValueError(f"Unsupported embedding model: {self.embedding_option}")
        except Exception as e:
            logger.error("Error initializing embedding model: %s", str(e), exc_info=True)
            raise EmbeddingGeneratorException(f"Error initializing embedding model: {str(e)}") from e

    def _initialize_vector_db(self):
        """Initialize vector database with appropriate configuration."""
        logger.debug("Initializing vector database: %s", self.vector_db_option)

        try:
            if self.vector_db_option == "Local (ChromaDB)":
                self.vector_db = ChromaVectorDatabase(
                    persist_directory=Config.CHROMA_PERSIST_DIR_PATH,
                    embedding_model=self.embedding_model
                )
            elif self.vector_db_option == "Pinecone":
                if not self.vector_db_api_key or not self.vector_db_index:
                    logger.error("Missing Pinecone credentials")
                    raise ValueError("Pinecone API key and index name are required")

                self.vector_db = PineconeVectorDatabase(
                    api_key=self.vector_db_api_key,
                    index_name=self.vector_db_index,
                    embedding_model=self.embedding_model
                )
            else:
                logger.error("Unsupported vector database type: %s", self.vector_db_option)
                raise ValueError(f"Unsupported vector database type: {self.vector_db_option}")
        except Exception as e:
            logger.error("Error initializing vector database: %s", str(e), exc_info=True)
            raise EmbeddingGeneratorException(f"Error initializing vector database: {str(e)}") from e

    def process_local_knowledge_base(self) -> Dict[str, Any]:
        """Process documents from local knowledge base."""
        logger.info("Processing local knowledge base")

        if not self.embedding_model:
            logger.error("Embedding model not configured")
            raise ValueError("Embedding model not configured")

        try:
            result = self._process_local_documents()
            logger.info("Successfully processed local knowledge base")
            return result
        except EmbeddingGeneratorException as e:
            logger.error("Error processing local knowledge base: %s", str(e), exc_info=True)
            raise EmbeddingGeneratorException(f"Error processing local knowledge base: {str(e)}") from e

    def process_uploaded_documents(self, files: List[Any]) -> Dict[str, Any]:
        """Process uploaded document files."""
        logger.info("Processing %d uploaded documents", len(files))
        
        try:
            documents = []
            doc_processor = DocumentProcessor()
            for file in files:
                logger.debug("Processing file: %s", file.name)
                content = doc_processor.read_file_content(file)
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": file.name,
                            "type": "uploaded_document"
                        }
                    )
                )
            
            self.vector_db.load_or_initialize(documents=[])
            self.vector_db.add_documents(documents)
            
            logger.info("Successfully processed %d documents", len(documents))
            return {
                "status": "success",
                "message": f"Successfully processed {len(documents)} documents"
            }
            
        except Exception as e:
            logger.error("Error processing uploaded documents: %s", str(e), exc_info=True)
            raise EmbeddingGeneratorException(f"Error processing uploaded documents: {str(e)}") from e

    def process_web_research(
        self,
        search_query: str,
        urls: List[str],
    ) -> Dict[str, Any]:
        """Process web research content."""
        logger.info("Processing web research for query: %s", search_query)
        
        try:
            sources = WebResearcher.perform_research(
                query=search_query,
                urls=urls,
                model_provider=self.web_research_provider,
                model_id=self.web_research_model_id,
                depth=1,
                api_key=self.web_research_provider_api_key
            )   
            
            documents = []
            for source in sources:
                url = source['url']
                content = source['content']

                logger.debug("Processing content from URL: %s", url)
                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "source": url,
                            "type": "web_research",
                            "query": search_query
                        }
                    )
                )
            
            self.vector_db.load_or_initialize(documents=[])
            self.vector_db.add_documents(documents)
            
            logger.info("Successfully processed %d web sources", len(documents))
            return {
                "status": "success",
                "message": f"Successfully processed {len(documents)} web sources"
            }
            
        except Exception as e:
            logger.error("Error processing web research: %s", str(e), exc_info=True)
            raise EmbeddingGeneratorException(f"Error processing web research: {str(e)}") from e

    def _process_local_documents(self) -> Dict[str, Any]:
        """Process documents for embedding generation and storage."""
        try:
            file_detector = FileChangeDetector(
                data_dir=Config.KNOWLEDGE_ARTICLES_DIR_PATH,
                metadata_file=Config.METADATA_FILE_PATH
            )

            new_or_updated_files = file_detector.detect_changes()
            valid_documents = file_detector.filter_empty_documents(new_or_updated_files)

            if not valid_documents:
                self.vector_db.load_or_initialize(documents=[])
                logger.info("No valid documents to process")
                #return {"status": "info", "message": "No valid documents to process"}
            else:
                self.vector_db.load_or_initialize(documents=valid_documents)
                self.vector_db.add_documents(valid_documents)

            deleted_files = file_detector.detect_deleted_files()
            if deleted_files:
                self.vector_db.delete_documents(deleted_files)
                logger.info("Removed %d deleted files", len(deleted_files))

            file_detector.save_metadata()
            logger.info("Processed %d documents and removed %d deleted files", 
                       len(valid_documents), len(deleted_files))
            return {
                "status": "success",
                "message": f"Processed {len(valid_documents)} documents and removed {len(deleted_files)} deleted files"
            }
            
        except (IOError, OSError) as e:
            # File system related errors
            logger.error("File system error while processing documents: %s", str(e), exc_info=True)
            raise EmbeddingGeneratorException(f"File system error: {str(e)}") from e
        except ValueError as e:
            # Data validation or format errors
            logger.error("Invalid document data: %s", str(e), exc_info=True)
            raise EmbeddingGeneratorException(f"Document validation error: {str(e)}") from e
        except (ImportError, ModuleNotFoundError) as e:
            # Module loading errors
            logger.error("Module loading error: %s", str(e), exc_info=True)
            raise EmbeddingGeneratorException(f"Module error: {str(e)}") from e
        except Exception as e:  # Fallback for unexpected errors
            logger.error("Unexpected error processing local documents: %s", str(e), exc_info=True)
            raise EmbeddingGeneratorException(f"Unexpected error: {str(e)}") from e

class EmbeddingGeneratorException(Exception):
    """Base exception for embedding generator related errors"""
    def __init__(self, message="Embedding generator error occurred"):
        self.message = message
        super().__init__(self.message)
