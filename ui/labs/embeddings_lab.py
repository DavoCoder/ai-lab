import os
from ui.labs.app_mode import AppMode
from config import Config
from embeddings.openai_embedding_model import OpenAIEmbeddingModel
from embeddings.huggingface_embedding_model import HuggingFaceEmbeddingModel
from vector_databases.chroma_vector_database import ChromaVectorDatabase
from vector_databases.pinecone_vector_database import PineconeVectorDatabase
from file_handler.file_change_detector import FileChangeDetector
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
class EmbeddingsLab(AppMode):

    @staticmethod
    def render():
        # Sidebar configuration
        st.sidebar.header("Embeddings Configuration")
        
        # Model Selection
        embedding_provider = st.sidebar.selectbox(
            "Select Embedding Provider",
            ["HuggingFace", "OpenAI"]
        )

        # Model-specific configuration
        embedding_model = None
        if embedding_provider == "HuggingFace":
            model_name = st.sidebar.selectbox(
                "Select HuggingFace Model",
                list(HuggingFaceEmbeddingModel.HUGGINGFACE_MODELS.keys()),
                format_func=lambda x: f"{x} ({HuggingFaceEmbeddingModel.HUGGINGFACE_MODELS[x]} dimensions)"
            )
            st.sidebar.info(f"Model dimensions: {HuggingFaceEmbeddingModel.HUGGINGFACE_MODELS[model_name]}")
            embedding_model = HuggingFaceEmbeddingModel(model_name=model_name).load_model()
        else:  # OpenAI
            #api_key = st.sidebar.text_input("OpenAI API Key", type="password")
            api_key = Config.OPENAI_API_KEY
            if api_key:
                embedding_model = OpenAIEmbeddingModel(api_key=api_key).load_model()

        # Vector Database Selection
        vector_db_type = st.sidebar.selectbox(
            "Select Vector Database",
            ["Local (ChromaDB)", "Pinecone"]
        )

        # Only show document processing for local DB
        if vector_db_type == "Local (ChromaDB)":
            st.subheader("Local Document Processing")
            
            tabs = st.tabs(["Document Processing", "Visualization"])
            
            with tabs[0]:
                if embedding_model:
                    if st.button("Process Documents"):
                        with st.spinner("Processing documents..."):

                            # Initialize vector DB
                            vector_db = ChromaVectorDatabase(
                                persist_directory=Config.CHROMA_PERSIST_DIR_PATH,
                                embedding_model=embedding_model
                            )
                            EmbeddingsLab._process_documents(vector_db)
                else:
                    st.warning("Please configure embedding model first.")
            
            with tabs[1]:
                # Add visualization if vector DB exists
                if os.path.exists(Config.CHROMA_PERSIST_DIR_PATH):
                    vector_db = ChromaVectorDatabase(
                        persist_directory=Config.CHROMA_PERSIST_DIR_PATH,
                        embedding_model=embedding_model
                    )
                    
                    EmbeddingsLab.visualize_embeddings(vector_db)
                else:
                    st.warning("No vector database found. Please process some documents first.")

        else:  # Pinecone
            st.subheader("Pinecone Configuration")
            #api_key = st.text_input("Pinecone API Key", type="password")
            api_key = Config.PINECONE_API_KEY
            #environment = st.text_input("Pinecone Environment")
            #index_name = st.text_input("Index Name")
            index_name = Config.PINECONE_INDEX_NAME
            
            # Picone implementation TBD
            if embedding_model:
                if st.button("Process Documents"):
                    with st.spinner("Processing documents..."):
                 
                        # Initialize vector DB
                        vector_db = PineconeVectorDatabase(
                            api_key=api_key,
                            index_name=index_name,
                            embedding_model=embedding_model
                        )

                        EmbeddingsLab._process_documents(vector_db)
            else:
                st.warning("Please configure embedding model first.")

        # visualization or analysis tools TBD
        st.subheader("Embedding Analysis Tools")
        # tools for visualizing and analyzing embeddings
        # For example:
        # - Dimensionality reduction plots
        # - Similarity search demos
        # - Embedding statistics

    @staticmethod
    def _process_documents(vector_db):
        """
        Process documents for embedding generation and storage
        
        Args:
            vector_db: Initialized vector database instance
        """
        try:
            # Initialize file detector
            file_detector = FileChangeDetector(
                data_dir=Config.KNOWLEDGE_ARTICLES_DIR_PATH,
                metadata_file=Config.METADATA_FILE_PATH
            )

            # Process new/updated files
            new_or_updated_files = file_detector.detect_changes()
            valid_documents = file_detector.filter_empty_documents(new_or_updated_files)

            if not valid_documents:
                st.info("No valid documents to process.")
                vector_db.load_or_initialize(documents=[])
            else:
                st.info(f"Found {len(valid_documents)} documents to process.")
                vector_db.load_or_initialize(documents=valid_documents)
                vector_db.add_documents(valid_documents)
                st.success(f"Successfully processed {len(valid_documents)} documents.")

            # Handle deleted files
            deleted_files = file_detector.detect_deleted_files()
            if deleted_files:
                st.info(f"Removing embeddings for {len(deleted_files)} deleted files...")
                vector_db.delete_documents(deleted_files)
                st.success("Deleted file embeddings removed from the vector database.")

            # Save metadata
            file_detector.save_metadata()
            st.success("Incremental update completed successfully!")
            
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
            raise
    
    @staticmethod
    def visualize_embeddings(vector_db):
        """Visualize embeddings using PCA and plotly"""
        st.subheader("Embeddings Visualization")
        
        try:
            # Initialize vector database first
            vector_db.load_or_initialize(documents=[])
            
            # Get all embeddings from ChromaDB
            results = vector_db.vector_db._collection.get(include=['embeddings', 'documents', 'metadatas'])
            
            # Check if we have any results
            if not isinstance(results, dict) or 'embeddings' not in results or len(results['embeddings']) == 0:
                st.warning("No embeddings found in the database.")
                return
                
            embeddings = np.array(results['embeddings'])
            documents = results.get('documents', [''] * len(embeddings))
            metadata = results.get('metadatas', [{'source': 'unknown'}] * len(embeddings))
            
            # Determine number of components based on data
            n_components = min(3, len(embeddings), embeddings.shape[1])
            
            # Reduce dimensions using PCA
            pca = PCA(n_components=n_components)
            reduced_embeddings = pca.fit_transform(embeddings)
            
            # Create DataFrame for plotting
            df = pd.DataFrame(
                reduced_embeddings,
                columns=[f'PC{i+1}' for i in range(n_components)]
            )
            df['document'] = documents
            # Safely handle metadata
            df['filename'] = [m.get('source', 'unknown') if m else 'unknown' for m in metadata]
            
            # Create scatter plot based on number of components
            if n_components == 3:
                fig = px.scatter_3d(
                    df,
                    x='PC1',
                    y='PC2',
                    z='PC3',
                    hover_data=['filename'],
                    title='Document Embeddings Visualization (PCA)',
                    labels={'PC1': 'First Component',
                            'PC2': 'Second Component',
                            'PC3': 'Third Component'}
                )
            elif n_components == 2:
                fig = px.scatter(
                    df,
                    x='PC1',
                    y='PC2',
                    hover_data=['filename'],
                    title='Document Embeddings Visualization (PCA)',
                    labels={'PC1': 'First Component',
                            'PC2': 'Second Component'}
                )
            else:
                st.warning("Not enough data for meaningful visualization")
                return
            
            st.plotly_chart(fig)
            
            # Show explained variance ratio
            explained_var = pca.explained_variance_ratio_
            st.info(f"Explained variance ratio: {explained_var.sum():.2%}")
            
            # Add similarity search demo
            st.subheader("Similarity Search Demo")
            search_query = st.text_input("Enter text to find similar documents:")
            if search_query:
                try:
                    similar_docs = vector_db.vector_db.similarity_search(search_query, k=3)
                    
                    if not similar_docs:
                        st.warning("No similar documents found.")
                    else:
                        st.write("Most similar documents:")
                        for i, doc in enumerate(similar_docs, 1):
                            st.markdown(f"**{i}. {doc.metadata.get('source', 'Unknown')}**")
                            st.write(doc.page_content[:200] + "...")
                            st.markdown("---")  # Add separator between documents
                        
                        # Optionally show number of documents found
                        st.info(f"Found {len(similar_docs)} similar documents.")
                        
                except Exception as e:
                    st.error(f"Error performing similarity search: {str(e)}")
            
        except Exception as e:
            st.error(f"Error visualizing embeddings: {str(e)}")
            raise
