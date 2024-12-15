from ui.labs.app_mode import AppMode
import streamlit as st

class ModelPlayground(AppMode):
    @staticmethod
    def render():
        st.sidebar.header("Model Settings")
        
        task_type = st.sidebar.selectbox(
            "Task Type",
            ["Text Generation", "Text Classification", "Named Entity Recognition", 
             "Summarization", "Translation"]
        )
        
        model_provider = st.sidebar.selectbox(
            "Model Provider",
            ["OpenAI", "Anthropic", "HuggingFace", "Custom"]
        )
        
        # Advanced Settings Expander
        with st.sidebar.expander("Advanced Settings"):
            temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
            max_tokens = st.slider("Max Tokens", 50, 2000, 500)
            top_p = st.slider("Top P", 0.0, 1.0, 1.0)