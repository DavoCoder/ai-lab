import os
from embeddings.openai_embedding_model import OpenAIEmbeddingModel
from vector_databases.chroma_vector_database import ChromaVectorDatabase
from file_handler.file_change_detector import FileChangeDetector

data_directory = os.getenv("KNOWLEDGE_ARTICLES_DIR_PATH")
metadata_file = os.getenv("METADATA_FILE_PATH")
persist_directory = os.getenv("CHROMA_PERSIST_DIR_PATH")

# Load API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OpenAI API key not found. Set the 'OPENAI_API_KEY' environment variable.")

# Create an Embedding Model
embedding_model_loader = OpenAIEmbeddingModel(api_key=openai_api_key)
embedding_model = embedding_model_loader.load_model()

# Use persistent storage for ChromaDB
vector_db = ChromaVectorDatabase(persist_directory=persist_directory, embedding_model=embedding_model)

#Create a file detector strategy
file_detector = FileChangeDetector(data_dir=data_directory, metadata_file=metadata_file)

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