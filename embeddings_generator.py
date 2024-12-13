import os
from embeddings.openai_embedding_model import OpenAIEmbeddingModel
from embeddings.huggingface_embedding_model import HuggingFaceEmbeddingModel
from vector_databases.chroma_vector_database import ChromaVectorDatabase
from file_handler.file_change_detector import FileChangeDetector
from config import Config

# Validate and set environment variables at startup
Config.validate()

# Create an Embedding Model
#embedding_model = OpenAIEmbeddingModel(api_key=openai_api_key).load_model()
embedding_model = HuggingFaceEmbeddingModel(model_name="all-MiniLM-L6-v2").load_model()

# Use persistent storage for ChromaDB
vector_db = ChromaVectorDatabase(persist_directory=Config.CHROMA_PERSIST_DIR_PATH, embedding_model=embedding_model)

#Create a file detector strategy
file_detector = FileChangeDetector(data_dir=Config.KNOWLEDGE_ARTICLES_DIR_PATH, metadata_file=Config.METADATA_FILE_PATH)

# Detect new or updated files
new_or_updated_files = file_detector.detect_changes()
# Filter out empty documents
valid_documents = file_detector.filter_empty_documents(new_or_updated_files)

# Exit if no valid documents
if not valid_documents:
    print("No valid documents to process. Exiting.")
    vector_db.load_or_initialize(documents=[])
else:
    # Load existing or initialize with empty documents
    vector_db.load_or_initialize(documents=valid_documents)
    # Add New Documents if Needed
    vector_db.add_documents(valid_documents)

# Detect and handle deleted files
deleted_files = file_detector.detect_deleted_files()
if deleted_files:
    print("Removing embeddings for deleted files...")
    vector_db.delete_documents(deleted_files)
    print("Deleted file embeddings removed from the vector database.")

# Save updated metadata
file_detector.save_metadata()
print("Incremental update completed successfully!")