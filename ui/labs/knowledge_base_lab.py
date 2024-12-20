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

# knowledge_base_lab.py

from pathlib import Path
import streamlit as st
from config import Config
from knowledge_base.local_knowledge_base import LocalKnowledgeBase

class KnowledgeBaseLab:
    """
    Knowledge Base Lab
    """
    @staticmethod
    def render():
        st.title("Knowledge Base Lab")
        
        # Create tabs for different operations
        tab_upload, tab_list, tab_delete = st.tabs([
            "Upload Documents", 
            "List Documents", 
            "Delete Documents"
        ])
        
        kb = LocalKnowledgeBase(Config.KNOWLEDGE_ARTICLES_DIR_PATH)

        with tab_upload:
            KnowledgeBaseLab._handle_upload_documents(kb)
            
        with tab_list:
            KnowledgeBaseLab._handle_list_documents(kb)
            
        with tab_delete:
            KnowledgeBaseLab._handle_delete_documents(kb)

    def _handle_upload_documents(kb: LocalKnowledgeBase):
        st.header("Upload Documents")
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files to upload", 
            accept_multiple_files=True,
            type=['txt', 'md', 'pdf', 'doc', 'docx']
        )
        
        # Optional subdirectory input
        subdirectory = st.text_input(
            "Subdirectory (optional)", 
            help="Enter a subdirectory path where files should be stored"
        )
        
        if uploaded_files and st.button("Upload Files"):
            for uploaded_file in uploaded_files:
                try:
                    content = uploaded_file.read()
                    if isinstance(content, bytes):
                        content = content.decode('utf-8')
                    
                    kb.create_file(
                        filename=uploaded_file.name,
                        content=content,
                        subdirectory=subdirectory if subdirectory else None
                    )
                    st.success(f"Successfully uploaded: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error uploading {uploaded_file.name}: {str(e)}")

    def _handle_list_documents(kb: LocalKnowledgeBase):
        st.header("Document List")
        # List all files recursively
        all_files = KnowledgeBaseLab._list_files_recursive(kb.base_path)
        
        if not all_files:
            st.info("No documents found in the knowledge base.")
            return
        
        # Display files in a table
        files_data = []
        for file_path in all_files:
            rel_path = file_path.relative_to(kb.base_path)
            files_data.append({
                "File": str(rel_path),
                "Size (bytes)": file_path.stat().st_size,
                "Directory": str(rel_path.parent) if str(rel_path.parent) != "." else "/"
            })
        
        st.dataframe(files_data)

    def _handle_delete_documents(kb: LocalKnowledgeBase):
        st.header("Delete Documents")
        
        # List all files for deletion
        all_files = KnowledgeBaseLab._list_files_recursive(kb.base_path)
        if not all_files:
            st.info("No documents to delete.")
            return
        
        # Create a multiselect for files
        file_options = [str(f.relative_to(kb.base_path)) for f in all_files]
        files_to_delete = st.multiselect(
            "Select files to delete",
            options=file_options
        )
        
        if files_to_delete and st.button("Delete Selected Files", type="primary"):
            for file_path in files_to_delete:
                try:
                    # Split the path to get subdirectory and filename
                    parts = Path(file_path).parts
                    if len(parts) > 1:
                        # If there's a subdirectory
                        filename = parts[-1]
                        subdirectory = str(Path(*parts[:-1]))
                        kb.delete_file(filename, subdirectory)
                    else:
                        # If file is in root directory
                        kb.delete_file(file_path)
                    st.success(f"Successfully deleted: {file_path}")
                except Exception as e:
                    st.error(f"Error deleting {file_path}: {str(e)}")
            
            # Rerun the app to refresh the file list
            st.rerun()

    def _list_files_recursive(directory: Path) -> list[Path]:
        """Recursively list all files in the directory."""
        files = []
        try:
            for item in directory.iterdir():
                if item.is_file():
                    files.append(item)
                elif item.is_dir():
                    files.extend(KnowledgeBaseLab._list_files_recursive(item))
        except Exception as e:
            st.error(f"Error accessing directory {directory}: {str(e)}")
        return files
        