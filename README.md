# AI-Powered RAG Pipeline

A Retrieval-Augmented Generation (RAG) system that combines document processing, vector storage, and large language models to provide accurate, context-aware responses to queries.

## 🌟 Features

- **Document Processing**: Support for various document formats
- **Vector Database Integration**: Efficient storage and retrieval of document embeddings
- **Query Pre-processing**: Advanced query rewriting and spell checking
- **Multiple LLM Support**: Integration with OpenAI's GPT models
- **FastAPI Backend**: RESTful API for easy integration
- **Modular Architecture**: Easy to extend and customize

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or 3.11
- OpenAI API key
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone [your-repo-url]
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
```bash
export OPENAI_API_KEY="your_api_key_here"
export KNOWLEDGE_ARTICLES_DIR_PATH="your_knowledge_articles_dir_path_here"
export CHROMA_PERSIST_DIR_PATH="your_chroma_persist_dir_path_here"         
export METADATA_FILE_PATH="your_metadata_file_path_here"
```

### Running the Application

#### FastAPI Backend
```bash
uvicorn main_api:app --reload
```

Access the API documentation at `http://localhost:8000/docs`

#### Streamlit Frontend (if applicable)
```bash
streamlit run app.py
```

## 🏗️ Project Structure

```
ai-playground/
├── embeddings/              # Embedding models
├── file_handler/            # File handling
├── knowledge_base/         # Document processing
├── query_pre_processing/   # Query enhancement
├── response_post_processing/   # Post-processing of the response
├── retrival_optimization/   # Optimization of the retrival
├── toxicity_detection/     # Toxicity detection
├── vector_databases/       # Vector storage
├── app.py                 # Streamlit frontend
├── main_api.py           # FastAPI backend
├── embeddings_generation.py # Embeddings generation
└── requirements.txt      # Project dependencies
```

## 🔧 Configuration

## 🚀 API Endpoints

- `POST /process-document`: Upload and process new documents
- `POST /query`: Query the system with natural language questions
- [Add other endpoints as applicable]

## 💡 Usage Examples

```python
# Example API call
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"query": "What is RAG?"}
)
print(response.json())
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

Built with ❤️ by DavoCoder