from dotenv import load_dotenv
import os
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

class Config:
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    
    # Paths
    CHROMA_PERSIST_DIR_PATH = os.getenv("CHROMA_PERSIST_DIR_PATH")
    KNOWLEDGE_ARTICLES_DIR_PATH = os.getenv("KNOWLEDGE_ARTICLES_DIR_PATH")
    METADATA_FILE_PATH = os.getenv("METADATA_FILE_PATH")
    # HuggingFace Settings
    TOKENIZERS_PARALLELISM = "false"

    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
    
    # Validation
    @classmethod
    def validate(cls):
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        if not cls.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
        
        # Ensure persist directory exists
        Path(cls.CHROMA_PERSIST_DIR_PATH).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def set_environment(cls):
        """Set necessary environment variables"""
        os.environ["OPENAI_API_KEY"] = cls.OPENAI_API_KEY
        os.environ["ANTHROPIC_API_KEY"] = cls.ANTHROPIC_API_KEY
        os.environ["CHROMA_PERSIST_DIR_PATH"] = cls.CHROMA_PERSIST_DIR_PATH
        os.environ["KNOWLEDGE_ARTICLES_DIR_PATH"] = cls.KNOWLEDGE_ARTICLES_DIR_PATH
        os.environ["METADATA_FILE_PATH"] = cls.METADATA_FILE_PATH
        os.environ["TOKENIZERS_PARALLELISM"] = cls.TOKENIZERS_PARALLELISM
