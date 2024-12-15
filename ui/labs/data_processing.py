from ui.labs.app_mode import AppMode
import streamlit as st

class DataProcessing(AppMode):
    @staticmethod
    def render():
        st.sidebar.header("Processing Options")
        
        process_type = st.sidebar.selectbox(
            "Process Type",
            ["Text Cleaning", "Document Splitting", "Format Conversion", 
             "Metadata Extraction", "Batch Processing"]
        )
        
        file_type = st.sidebar.multiselect(
            "File Types",
            ["PDF", "TXT", "DOCX", "HTML", "JSON"]
        )