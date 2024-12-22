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

# rag_processor_api.py
from enum import Enum
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from rag.rag_processor import RAGProcessor

app = FastAPI(
    title="RAG Processing API",
    description="REST API for Retrieval Augmented Generation operations",
    version="1.0.0",
    openapi_tags=[{
        "name": "rag",
        "description": "Endpoints for RAG processing operations"
    }]
)

class LLMOption(str, Enum):
    GPT4 = "OpenAI GPT-4"
    GPT35 = "OpenAI GPT-3.5"
    CLAUDE3_OPUS = "Anthropic Claude-3 Opus"
    CLAUDE3_SONNET = "Anthropic Claude-3 Sonnet"

class EmbeddingOption(str, Enum):
    OPENAI = "OpenAI"
    HUGGINGFACE = "HuggingFace"

class VectorDBOption(str, Enum):
    CHROMADB = "ChromaDB"
    PINECONE = "Pinecone"

class ToxicityOption(str, Enum):
    OPENAI = "OpenAI"
    HUGGINGFACE = "HuggingFace"

class ProcessOption(str, Enum):
    SPELL_CHECK = "Spell Check"
    QUERY_REWRITER = "Query Rewriter"
    HALLUCINATION_FILTER = "Hallucination Filter"
    SUMMARIZATION = "Summarization"

class RAGConfig(BaseModel):
    llm_option: LLMOption = Field(..., description="The LLM model to use for generation")
    llm_api_key: str = Field(..., description="API key for the selected LLM")
    llm_settings: Dict[str, Any] = Field(
        default_factory=lambda: {"temperature": 0.7, "max_tokens": 500},
        description="LLM-specific settings like temperature and max tokens"
    )
    embedding_option: EmbeddingOption = Field(..., description="The embedding model to use")
    embedding_api_key: str = Field(..., description="API key for the embedding service")
    vector_db_option: VectorDBOption = Field(..., description="Vector database selection")
    vector_db_api_key: str = Field(..., description="API key for the vector database")
    vector_db_index: str = Field(..., description="Index name in the vector database")
    toxicity_option: ToxicityOption = Field(..., description="Toxicity detection service")
    toxicity_api_key: str = Field(..., description="API key for toxicity detection")
    pre_process_options: List[ProcessOption] = Field(
        default=[],
        description="List of pre-processing steps to apply"
    )
    post_process_options: List[ProcessOption] = Field(
        default=[],
        description="List of post-processing steps to apply"
    )

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user query to process")

class QueryResponse(BaseModel):
    response: str = Field(..., description="The processed response")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata about the processing"
    )

def get_rag_processor(config: RAGConfig) -> RAGProcessor:
    return RAGProcessor(
        llm_provider=config.llm_provider,
        llm_model=config.llm_model,
        llm_api_key=config.llm_api_key,
        llm_settings=config.llm_settings,
        embedding_option=config.embedding_option,
        embedding_api_key=config.embedding_api_key,
        vector_db_option=config.vector_db_option,
        vector_db_api_key=config.vector_db_api_key,
        vector_db_index=config.vector_db_index,
        toxicity_option=config.toxicity_option,
        toxicity_api_key=config.toxicity_api_key,
        pre_process_options=config.pre_process_options,
        post_process_options=config.post_process_options,
    )

@app.post(
    "/api/v1/rag",
    response_model=QueryResponse,
    tags=["rag"],
    summary="Process a query using RAG",
    description="Processes a user query through the RAG pipeline with the specified configuration"
)
async def process_query(
    query_request: QueryRequest,
    config: RAGConfig,
) -> QueryResponse:
    try:
        processor = get_rag_processor(config)
        response = processor.initialize_and_execute_all(query_request.query)
        
        return QueryResponse(
            response=response,
            metadata={
                "llm_used": config.llm_option,
                "pre_processing_steps": config.pre_process_options,
                "post_processing_steps": config.post_process_options
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        ) from e

@app.get(
    "/api/v1/rag/options",
    tags=["rag"],
    summary="Get available RAG options",
    description="Returns all available options for configuring the RAG pipeline"
)
async def get_options():
    return {
        "llm_options": [e.value for e in LLMOption],
        "embedding_options": [e.value for e in EmbeddingOption],
        "vector_db_options": [e.value for e in VectorDBOption],
        "toxicity_options": [e.value for e in ToxicityOption],
        "processing_options": [e.value for e in ProcessOption]
    }

@app.get(
    "/health",
    tags=["health"],
    summary="Health check",
    description="Check if the API is running"
)
async def health_check():
    return {"status": "healthy"}
