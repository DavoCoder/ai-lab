from ui.labs.app_mode import AppMode
import streamlit as st
from typing import Dict, Any
from nlp_processing.nlp_processor import NLPProcessor
class ModelPlayground(AppMode):
    # Model mappings for different providers and tasks
    @staticmethod
    def render():
        st.sidebar.header("Model Settings")
        
        # Task Selection
        task_type = st.sidebar.selectbox(
            "Task Type",
            list(NLPProcessor.TASK_DESCRIPTIONS.keys())
        )
        st.sidebar.caption(NLPProcessor.TASK_DESCRIPTIONS[task_type])
        
        # Provider Selection
        model_provider = st.sidebar.selectbox(
            "Model Provider",
            list(NLPProcessor.PROVIDER_MODELS.keys())
        )
        
        # Model Selection based on provider and task
        available_models = NLPProcessor.PROVIDER_MODELS[model_provider][task_type]["models"]
        selected_model = st.sidebar.selectbox("Model", available_models)

        model_id = NLPProcessor.PROVIDER_MODELS[model_provider][task_type]["model_ids"][selected_model]
        
        # Provider-specific settings
        api_key = None
        if model_provider in ["OpenAI", "Anthropic"]:
            api_key = st.sidebar.text_input(f"{model_provider} API Key", type="password")
        elif model_provider == "Custom":
            model_path = st.sidebar.text_input("Model Path/URL")
        
        # Advanced Settings
        with st.sidebar.expander("Advanced Settings"):
            settings = ModelPlayground._render_advanced_settings(task_type)
        
        # Main content area
        st.title(f"🤖 {task_type}")
        
        # Input area based on task
        input_text = ModelPlayground._render_input_area(task_type)
        
        # Process button
        if st.button("Process"):
            if not api_key and model_provider in ["OpenAI", "Anthropic"]:
                st.error(f"Please provide {model_provider} API key")
                return
                
            with st.spinner("Processing..."):
                try:
                    result = NLPProcessor.process_task(
                        task_type=task_type,
                        model_provider=model_provider,
                        model=model_id,
                        input_text=input_text,
                        settings=settings,
                        api_key=api_key
                    )
                    ModelPlayground._display_results(task_type, result)
                except Exception as e:
                    st.error(f"Error processing task: {str(e)}")

    @staticmethod
    def _render_advanced_settings(task_type: str) -> Dict[str, Any]:
        settings = {}
        # Add default settings
        for setting_name, setting_config in NLPProcessor.TASK_SETTINGS["default"].items():
            if setting_config["type"] in ["float", "int"]:
                settings[setting_name] = st.slider(
                    setting_name.replace("_", " ").title(),
                    setting_config["min"],
                    setting_config["max"],
                    setting_config["default"]
                )
        
        # Add task-specific settings
        if task_type in NLPProcessor.TASK_SETTINGS:
            for setting_name, setting_config in NLPProcessor.TASK_SETTINGS[task_type].items():
                if setting_config["type"] in ["float", "int"]:
                    settings[setting_name] = st.slider(
                        setting_name.replace("_", " ").title(),
                        setting_config["min"],
                        setting_config["max"],
                        setting_config["default"]
                    )
                elif setting_config["type"] == "select":
                    settings[setting_name] = st.selectbox(
                        setting_name.replace("_", " ").title(),
                        setting_config["options"],
                        index=setting_config["options"].index(setting_config["default"])
                    )
        
        return settings

    @staticmethod
    def _render_input_area(task_type: str) -> str:
        if task_type == "Text Generation":
            return st.text_area("Enter your prompt:", height=100)
        elif task_type == "Text Classification":
            return st.text_area("Enter text to classify:", height=100)
        elif task_type == "Named Entity Recognition":
            return st.text_area("Enter text for NER analysis:", height=100)
        elif task_type == "Summarization":
            return st.text_area("Enter text to summarize:", height=200)
        elif task_type == "Translation":
            return st.text_area("Enter text to translate:", height=100)

    @staticmethod
    def _display_results(task_type: str, result):
        st.subheader("Results")
        
        if task_type == "Text Classification":
            st.json(result)
        elif task_type == "Named Entity Recognition":
            for entity in result:
                st.markdown(f"**{entity['text']}** - {entity['label']}")
        else:
            st.write(result)
            
        # Add copy button for results
        st.button("📋 Copy Results")