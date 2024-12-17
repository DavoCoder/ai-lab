from ui.home import Home
from ui.labs.rag_pipeline import RAGPipeline
from ui.labs.embeddings_lab import EmbeddingsLab
from ui.labs.web_research import WebResearch
from ui.labs.model_playground import ModelPlayground
from ui.labs.data_processing import DataProcessing
from ui.labs.knowledge_base_lab import KnowledgeBaseLab
# Title and description mapping
MODE_INFO = {
    "Home": {
        "title": "🏠 AI Lab Home",
        "description": "Welcome to the AI Lab - Your playground for AI experimentation"
    },
    "Model Playground": {
        "title": "🤖 Model Playground",
        "description": "Experiment with different AI models and tasks in an interactive environment."
    },
    "Web Research": {
        "title": "🌐 Web Research Assistant",
        "description": "Automated research tools for gathering and synthesizing information from various web sources."
    },
    "Data Processing": {
        "title": "⚙️ Data Processing Hub",
        "description": "Tools for cleaning, transforming, and preparing your data for AI applications."
    },
    "Knowledge Base": {
        "title": "⚙️ Knowledge Base Lab",
        "description": "Tools for creating and deleting files in a local knowledge base."
    },
    "Embeddings Lab": {
        "title": "🧬 Embeddings Laboratory",
        "description": "Experiment with different embedding models, compare and visualize vector representations."
    },
    "RAG Pipeline": {
        "title": "🔍 RAG Pipeline",
        "description": "Retrieval-Augmented Generation system for enhanced question answering using your documents."
    }
}

# Mode mapping
MODE_CLASSES = {
    "Home": Home,
    "Model Playground": ModelPlayground,
    "Web Research": WebResearch,
    "Data Processing": DataProcessing,
    "Knowledge Base": KnowledgeBaseLab,
    "Embeddings Lab": EmbeddingsLab,
    "RAG Pipeline": RAGPipeline 
}