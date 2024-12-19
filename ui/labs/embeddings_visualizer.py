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

# embeddings_visualizer.py
import streamlit as st
import plotly.express as px
from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

class EmbeddingsVisualizer:
    
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