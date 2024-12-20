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

# local_knowledge_base.py
import shutil
from pathlib import Path
from typing import Optional

class LocalKnowledgeBase:
    def __init__(self, base_path: str):
        """Initialize the local knowledge base with a base directory path.
        
        Args:
            base_path (str): The root directory for the knowledge base
        """
        self.base_path = Path(base_path)
        self._ensure_base_directory()

    def _ensure_base_directory(self) -> None:
        """Create the base directory if it doesn't exist."""
        self.base_path.mkdir(parents=True, exist_ok=True)

    def create_file(self, filename: str, content: str, subdirectory: Optional[str] = None) -> Path:
        """Create a new file in the knowledge base.
        
        Args:
            filename (str): Name of the file to create
            content (str): Content to write to the file
            subdirectory (Optional[str]): Optional subdirectory path within the base directory
            
        Returns:
            Path: Path object pointing to the created file
            
        Raises:
            FileExistsError: If the file already exists
        """
        if subdirectory:
            file_path = self.base_path / subdirectory / filename
            (self.base_path / subdirectory).mkdir(parents=True, exist_ok=True)
        else:
            file_path = self.base_path / filename

        if file_path.exists():
            raise FileExistsError(f"File {file_path} already exists")

        file_path.write_text(content, encoding='utf-8')
        return file_path

    def delete_file(self, filename: str, subdirectory: Optional[str] = None) -> None:
        """Delete a file from the knowledge base.
        
        Args:
            filename (str): Name of the file to delete
            subdirectory (Optional[str]): Optional subdirectory path within the base directory
            
        Raises:
            FileNotFoundError: If the file doesn't exist
        """
        if subdirectory:
            file_path = self.base_path / subdirectory / filename
        else:
            file_path = self.base_path / filename

        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} does not exist")

        if file_path.is_file():
            file_path.unlink()
        else:
            shutil.rmtree(file_path)

    def delete_directory(self, directory: str) -> None:
        """Delete a directory and all its contents from the knowledge base.
        
        Args:
            directory (str): Directory path to delete
            
        Raises:
            FileNotFoundError: If the directory doesn't exist
        """
        dir_path = self.base_path / directory
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory {dir_path} does not exist")

        shutil.rmtree(dir_path)
