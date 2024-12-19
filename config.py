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

# config.py
from dotenv import load_dotenv
import os
from pathlib import Path
import json
from typing import Dict, Any

# Load environment variables from .env file
load_dotenv()

class Config:
    CONFIG_DIR = Path(__file__).parent / "config"

    # Knowledge base paths
    KNOWLEDGE_ARTICLES_DIR_PATH = os.getenv("KNOWLEDGE_ARTICLES_DIR_PATH")
    METADATA_FILE_PATH = os.getenv("METADATA_FILE_PATH")

    # HuggingFace Settings
    TOKENIZERS_PARALLELISM = "false"

    # Chroma Settings
    CHROMA_PERSIST_DIR_PATH = os.getenv("CHROMA_PERSIST_DIR_PATH")

    # Google Auth Settings
    GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
    GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")

    # Auth Database paths
    AUTH_DB_PATH = "users.db"
    AUTH_DB_SCHEMA_PATH = "auth_schema.sql"
    
    # Validation
    @classmethod
    def validate(cls):
        # Ensure persist directory exists
        Path(cls.CHROMA_PERSIST_DIR_PATH).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def set_environment(cls):
        """Set necessary environment variables"""
        os.environ["CHROMA_PERSIST_DIR_PATH"] = cls.CHROMA_PERSIST_DIR_PATH
        os.environ["KNOWLEDGE_ARTICLES_DIR_PATH"] = cls.KNOWLEDGE_ARTICLES_DIR_PATH
        os.environ["METADATA_FILE_PATH"] = cls.METADATA_FILE_PATH
        os.environ["TOKENIZERS_PARALLELISM"] = cls.TOKENIZERS_PARALLELISM

    @classmethod
    def load_config(cls, filename: str) -> Dict[str, Any]:
        """Load configuration from JSON file."""
        config_path = cls.CONFIG_DIR / filename
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"Error loading configuration {filename}: {str(e)}")

    @classmethod
    def load_all_configs(cls) -> Dict[str, Dict[str, Any]]:
        """Load all configuration files."""
        try:
            return {
                "provider_models": cls.load_config("provider_models.json"),
                "task_descriptions": cls.load_config("task_descriptions.json"),
                "task_settings": cls.load_config("task_settings.json"),
                "system_prompts": cls.load_config("system_prompts.json")
            }
        except Exception as e:
            raise Exception(f"Error loading configurations: {str(e)}")
