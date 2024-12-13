import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
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
from config import Config

# Validate and set environment variables at startup
Config.validate()

# Initialize FastAPI
app = FastAPI(title="RAG Pipeline API", description="An API to expose the RAG pipeline for querying documents.", version="1.0")

# Define Request Schema
class QueryRequest(BaseModel):
    query: str
    llm: str  # e.g., "OpenAI GPT-4"
    embedding_model: str  # "OpenAI" or "HuggingFace"
    vector_db: str  # "ChromaDB" or "Pinecone"
    preprocessing_steps: Optional[List[str]] = []  # ["Spell Check", "Query Rewriter"]
    postprocessing_steps: Optional[List[str]] = []  # ["Hallucination Filter", "Summarization"]
    toxicity_detection: Optional[str] = None  # "None", "OpenAI", "HuggingFace"

# Root Endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG Pipeline API!"}

# Query Endpoint
@app.post("/query")
def process_query(request: QueryRequest):
    try:
        # Initialize Embedding Model
        if request.embedding_model == "OpenAI":
            embedding_model = OpenAIEmbeddingModel(api_key=Config.OPENAI_API_KEY).load_model()
        elif request.embedding_model == "HuggingFace":
            embedding_model = HuggingFaceEmbeddingModel(model_name="all-MiniLM-L6-v2").load_model()
        else:
            raise HTTPException(status_code=400, detail="Invalid embedding model selected.")
        
        # Initialize Vector Database
        if request.vector_db == "ChromaDB":
            vector_db = ChromaVectorDatabase(persist_directory=Config.CHROMA_PERSIST_DIR_PATH, embedding_model=embedding_model)
        elif request.vector_db == "Pinecone":
            vector_db = PineconeVectorDatabase(
                index_name="rag-index", 
                embedding_model=embedding_model,
                api_key="your-pinecone-api-key", 
                environment="your-pinecone-env"
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid vector database selected.")
        
        # Initialize Toxicity Detector
        toxicity_detector = None
        if request.toxicity_detection == "OpenAI":
            toxicity_detector = OpenAIToxicityDetector(api_key=Config.OPENAI_API_KEY)
        elif request.toxicity_detection == "HuggingFace":
            toxicity_detector = HuggingFaceToxicityDetector()

        # Apply Pre-Processing Steps
        query = request.query
        for step in request.preprocessing_steps:
            if step == "Spell Check":
                processor = SpellCheckQueryProcessor()
                query = processor.process(query)
            elif step == "Query Rewriter":
                processor = QueryRewriterProcessor()
                query = processor.process(query)

        # Toxicity Detection
        if toxicity_detector and toxicity_detector.detect_toxicity(query):
            return {"response": "Query contains toxic content. Aborting."}

        # Retrieval and Response Generation
        vector_db.load_or_initialize(documents=[])
        retriever = vector_db.get_retriever(k=3)
        llm = ChatOpenAI(model="gpt-4" if request.llm == "OpenAI GPT-4" else "gpt-3.5-turbo", temperature=0)
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Retrieve and Generate Response
        response = qa_chain.run(query)

        # Apply Post-Processing Steps
        for step in request.postprocessing_steps:
            if step == "Hallucination Filter":
                post_processor = HallucinationFilter()
                response = post_processor.process(response, retriever.get_relevant_documents(query))
            elif step == "Summarization":
                post_processor = SummarizationPostProcessor()
                response = post_processor.process(response, None)

        return {"response": response}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
