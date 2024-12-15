from ui.labs.app_mode import AppMode
import streamlit as st

class WebResearch(AppMode):
    @staticmethod
    def render():
        st.sidebar.header("Research Configuration")
        
        search_type = st.sidebar.selectbox(
            "Research Type",
            ["Web Search", "Academic Papers", "News Articles", "Custom Sources"]
        )
        
        depth_level = st.sidebar.slider("Search Depth", 1, 5, 2)
        
        include_citations = st.sidebar.checkbox("Include Citations", value=True)