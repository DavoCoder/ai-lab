FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy all files into the working directory
COPY . /app

# Create two directories for temporary file storage
RUN mkdir -p /app/knowledge_articles /app/chroma_db
# Set appropriate permissions for the directories
RUN chmod 777 /app/knowledge_articles /app/chroma_db

# Create a .env file with 3 specific paths
RUN echo "CHROMA_PERSIST_DIR_PATH=/app/chroma_db/" > .env && \
    echo "KNOWLEDGE_ARTICLES_DIR_PATH=/app/knowledge_articles/" >> .env && \
    echo "METADATA_FILE_PATH=/app/metadata.json" >> .env

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port Streamlit will run on
EXPOSE 8080

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]