from ui.home import Home
from ui.labs.rag_pipeline import RAGPipeline
from ui.labs.embeddings_lab import EmbeddingsLab
from ui.labs.web_research import WebResearch
from ui.labs.model_playground import ModelPlayground
from ui.labs.data_processing import DataProcessing

# Title and description mapping
MODE_INFO = {
    "Home": {
        "title": "🏠 AI Lab Home",
        "description": "Welcome to the AI Lab - Your playground for AI experimentation"
    },
    "RAG Pipeline": {
        "title": "🔍 RAG Pipeline",
        "description": "Retrieval-Augmented Generation system for enhanced question answering using your documents."
    },
    "Embeddings Lab": {
        "title": "🧬 Embeddings Laboratory",
        "description": "Experiment with different embedding models, compare and visualize vector representations."
    },
    "Web Research": {
        "title": "🌐 Web Research Assistant",
        "description": "Automated research tools for gathering and synthesizing information from various web sources."
    },
    "Model Playground": {
        "title": "🤖 Model Playground",
        "description": "Experiment with different AI models and tasks in an interactive environment."
    },
    "Data Processing": {
        "title": "⚙️ Data Processing Hub",
        "description": "Tools for cleaning, transforming, and preparing your data for AI applications."
    }
}

# Mode mapping
MODE_CLASSES = {
     "Home": Home,
    "RAG Pipeline": RAGPipeline,
    "Embeddings Lab": EmbeddingsLab,
    "Web Research": WebResearch,
    "Model Playground": ModelPlayground,
    "Data Processing": DataProcessing
}