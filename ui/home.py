from ui.labs.app_mode import AppMode
import streamlit as st
from pathlib import Path

class Home(AppMode):
    @staticmethod
    def render():
        # Get the path to the markdown file
        docs_path = Path(__file__).parent / "home.md"
        
        try:
            # Read the markdown content
            with open(docs_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Render the markdown
            st.markdown(content, unsafe_allow_html=True)
            
        except FileNotFoundError:
            st.error("Documentation file not found. Please check if docs/home.md exists.")
        except Exception as e:
            st.error(f"Error loading documentation: {str(e)}")
