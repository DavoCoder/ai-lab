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

# mode_config.py
from ui.home import Home
from ui.labs.rag_pipeline import RAGPipeline
from ui.labs.embeddings_lab import EmbeddingsLab
from ui.labs.web_research import WebResearch
from ui.labs.model_playground import ModelPlayground
from ui.labs.data_processing import DataProcessing
from ui.labs.knowledge_base_lab import KnowledgeBaseLab
from ui.labs.agent_lab import AgentLab
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
    },
    "Agent Lab": {
        "title": "🤖 Agent Lab",
        "description": "Experiment with different AI agents for research, document analysis, and code assistance."
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
    "RAG Pipeline": RAGPipeline,
    "Agent Lab": AgentLab
}
