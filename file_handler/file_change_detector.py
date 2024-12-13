import os
import hashlib
import json
from langchain_community.document_loaders import DirectoryLoader, TextLoader

class FileChangeDetector:
    def __init__(self, data_dir, metadata_file="metadata.json"):
        """
        Initialize the FileChangeDetector.

        Args:
            data_dir (str): Directory containing the documents to monitor.
            metadata_file (str): Path to the metadata file storing file hashes.
        """
        self.data_dir = data_dir
        self.metadata_file = metadata_file
        self.processed_files = self._load_metadata()

    def _load_metadata(self):
        """Load existing metadata or create an empty dictionary."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, "r") as f:
                return json.load(f)
        return {}

    @staticmethod
    def _get_file_hash(filepath):
        """Generate a SHA-256 hash for a file."""
        with open(filepath, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def detect_changes(self):
        """
        Detect new or updated files.

        Returns:
            list: List of LangChain document objects representing new or updated files.
        """
        # Load all documents from the data directory
        loader = DirectoryLoader(self.data_dir, glob="**/*.txt", loader_cls=TextLoader)
        all_documents = loader.load()

        print(f"Loaded {len(all_documents)} documents from '{self.data_dir}'.")

        # Detect new or updated files
        new_or_updated_files = []
        for doc in all_documents:
            file_path = doc.metadata['source']
            file_hash = self._get_file_hash(file_path)

            if file_path not in self.processed_files or self.processed_files[file_path] != file_hash:
                new_or_updated_files.append(doc)
                self.processed_files[file_path] = file_hash

        print(f"New or updated documents: {[doc.metadata['source'] for doc in new_or_updated_files]}")
        return new_or_updated_files

    def detect_deleted_files(self):
        """
        Detect files that have been deleted since the last run.

        Returns:
            list: List of file paths that were deleted.
        """
        deleted_files = [file for file in self.processed_files if not os.path.exists(file)]

        if deleted_files:
            print(f"Detected deleted files: {deleted_files}")
            # Remove deleted files from the metadata
            for file in deleted_files:
                del self.processed_files[file]

        return deleted_files

    def save_metadata(self):
        """Save the updated metadata to the metadata file."""
        with open(self.metadata_file, "w") as f:
            json.dump(self.processed_files, f, indent=4)
        print("Metadata saved successfully.")

    @staticmethod
    def filter_empty_documents(documents):
        """
        Filter out empty or invalid documents.

        Args:
            documents (list): List of LangChain document objects.

        Returns:
            list: List of valid (non-empty) documents.
        """
        valid_documents = [doc for doc in documents if doc.page_content.strip()]
        if not valid_documents:
            print("All documents are empty or invalid. Nothing to process.")
        return valid_documents
