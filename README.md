# RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that combines document processing, vector storage, and large language models to provide accurate, context-aware responses to queries.

## 🌟 Features

- **Document Processing**: Support for various document formats
- **Vector Database Integration**: Efficient storage and retrieval of document embeddings
- **Query Pre-processing**: Advanced query rewriting and spell checking
- **Multiple LLM Support**: Integration with OpenAI's GPT models
- **FastAPI Backend**: RESTful API for easy integration
- **Modular Architecture**: Easy to extend and customize

## 🚀 Getting Started

### Minimum Prerequisites

- Python 3.10 or 3.11
- OpenAI API key
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/DavoCoder/ai-playground.git
cd ai-playground
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Unix/macOS
# or
venv\Scripts\activate     # On Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:

- Create a .env file in the root folder
- Use the .env_example file to get the environment variables needed
- Fill out the environment variables with your local paths
- CHROMA_PERSIST_DIR_PATH is a local directory path to create the ChromaDB
- KNOWLEDGE_ARTICLES_DIR_PATH is a local directory path to get Documents for creating embeddings for the ChromaDB. Files supported .txt
- METADATA_FILE_PATH is a local file path to store hash maps for identifying changes in the knowledge base article dir and files

### Running the Application

#### Streamlit Frontend
```bash
streamlit run app.py
```

#### FastAPI Backend
```bash
uvicorn rest_api.rag_processor_api:app --reload
```

Access the API documentation at `http://localhost:8000/docs`

## 🏗️ Project Structure

```
ai-playground/
├── config/                  # Configuration
├── data_processing/         # Data processing
├── embeddings/              # Embedding models
├── file_handler/            # File handling
├── knowledge_base/         # Document processing
├── nlp_processing/         # NLP processing models
├── query_pre_processing/   # Query enhancement
├── rag/                    # RAG processing
├── response_post_processing/   # Post-processing of the response
├── rest_api/               # REST API (FastAPI)
├── retrival_optimization/   # Optimization of the retrival
├── toxicity_detection/     # Toxicity detection
├── ui/                     # UI
├── vector_databases/       # Vector storage
├── web_research/           # Web research
├── app.py                 # Streamlit frontend
├── embeddings_generation.py # Embeddings generation
└── requirements.txt      # Project dependencies
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📝 License

TBD

## 🙏 Acknowledgments

- OpenAI for LLM support
- LangChain for the framework

## 📞 Contact

DavoCoder

---

Built by DavoCoder