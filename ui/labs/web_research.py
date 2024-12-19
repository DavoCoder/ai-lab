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

# web_research.py
from ui.labs.app_mode import AppMode
from typing import Tuple
import streamlit as st
from web_research.web_researcher import WebResearcher
from nlp_processing.nlp_processor import NLPProcessor


class WebResearch(AppMode):
    @staticmethod
    def render():
        st.sidebar.header("Research Configuration")
        
        # Research Configuration
        search_type = st.sidebar.selectbox(
            "Research Type",
            ["Custom Sources / URLs", "Web Search", "Academic Papers", "News Articles"]
        )
        
        depth_level = st.sidebar.slider("Search Depth", 1, 5, 2)
        include_citations = st.sidebar.checkbox("Include Citations", value=True)

        # Model Selection for Research
        st.sidebar.subheader("Research Model")
        research_provider = st.sidebar.selectbox(
            "Research Provider",
            ["OpenAI", "Anthropic"],
            key="research_provider"
        )
        
        research_models = NLPProcessor.PROVIDER_MODELS[research_provider]["Text Generation"]["models"]
        research_model = st.sidebar.selectbox(
            "Research Model",
            research_models,
            key="research_model",
            help="Model used for content evaluation and extraction"
        )

         # API Keys
        research_api_key = None
        if research_provider in ["OpenAI", "Anthropic"]:
            research_api_key = st.sidebar.text_input(
                f"{research_provider} Research API Key",
                type="password",
                help=f"API key for {research_provider} research model"
            )

        # Model Selection for Synthesis
        st.sidebar.subheader("Synthesis Model")
        synthesis_provider = st.sidebar.selectbox(
            "Synthesis Provider",
            ["OpenAI", "Anthropic"],
            key="synthesis_provider"
        )
        
        synthesis_models = NLPProcessor.PROVIDER_MODELS[synthesis_provider]["Summarization"]["models"]
        synthesis_model = st.sidebar.selectbox(
            "Synthesis Model",
            synthesis_models,
            key="synthesis_model",
            help="Model used for synthesizing research findings"
        )

        synthesis_api_key = None
        if synthesis_provider in ["OpenAI", "Anthropic"]:
            synthesis_api_key = st.sidebar.text_input(
                f"{synthesis_provider} Synthesis API Key",
                type="password",
                help=f"API key for {synthesis_provider} synthesis model"
            )

        # Main content area
        st.header("Web Research Assistant")
        
        # Research query input
        research_query = st.text_area(
            "Research Query",
            height=100,
            help="Enter your research question or topic"
        )

        if search_type == "Custom Sources / URLs":
            # Get URLs from user input
            urls = st.text_area(
                "Enter URLs (one per line)",
                height=100
            ).strip().split('\n')
            
            urls = [url.strip() for url in urls if url.strip()]
            
            if not urls:
                st.warning("Please enter at least one URL.")
                return []
                
        else:
            st.warning("Other search types not yet implemented")
            return []
        
        if st.button("Start Research"):
            # Validate all settings before proceeding
            is_valid, error_message = WebResearch._validate_model_settings(
                research_provider=research_provider,
                research_model=research_model,
                synthesis_provider=synthesis_provider,
                synthesis_model=synthesis_model,
                research_api_key=research_api_key,
                synthesis_api_key=synthesis_api_key
            )
            
            if not is_valid:
                st.error(error_message)
                return
                
            if not research_query:
                st.error("Please enter a research query")
                return
                
                
            try:
                with st.spinner("Researching..."):
                    # Initialize models with their full IDs
                    research_model_id = NLPProcessor.PROVIDER_MODELS[research_provider]["Text Generation"]["model_ids"][research_model]
                    synthesis_model_id = NLPProcessor.PROVIDER_MODELS[synthesis_provider]["Summarization"]["model_ids"][synthesis_model]
                    
                    # Perform research
                    sources = WebResearcher.perform_research(
                        query=research_query,
                        urls=urls,
                        model_provider=research_provider,
                        model_id=research_model_id,
                        depth=depth_level,
                        api_key=research_api_key
                    )
                    
                    # Synthesize findings
                    synthesis = WebResearcher.synthesize_research(
                        sources=sources,
                        query=research_query,
                        include_citations=include_citations,
                        model_provider=synthesis_provider,
                        model_id=synthesis_model_id,
                        api_key=synthesis_api_key
                    )
                    
                    # Display results
                    st.subheader("Research Synthesis")
                    st.write(synthesis)

                    st.subheader("Research Results")
                    st.write(sources)
                    
                    if include_citations:
                        st.subheader("Sources")
                        for idx, source in enumerate(sources, 1):
                            st.markdown(f"{idx}. [{source['title']}]({source['url']})")
                            
            except Exception as e:
                st.error(f"An error occurred during research: {str(e)}")

    @staticmethod
    def _validate_model_settings(
            research_provider: str,
            research_model: str,
            synthesis_provider: str,
            synthesis_model: str,
            research_api_key: str = None,
            synthesis_api_key: str = None
        ) -> Tuple[bool, str]:
        """
        Validates model settings and API keys.
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Check if providers are supported
        supported_providers = ["OpenAI", "Anthropic"]
        if research_provider not in supported_providers:
            return False, f"Unsupported research provider: {research_provider}"
        if synthesis_provider not in supported_providers:
            return False, f"Unsupported synthesis provider: {synthesis_provider}"
            
        # Validate research model exists for provider
        if research_model not in NLPProcessor.PROVIDER_MODELS[research_provider]["Text Generation"]["models"]:
            return False, f"Invalid research model {research_model} for provider {research_provider}"
            
        # Validate synthesis model exists for provider
        if synthesis_model not in NLPProcessor.PROVIDER_MODELS[synthesis_provider]["Summarization"]["models"]:
            return False, f"Invalid synthesis model {synthesis_model} for provider {synthesis_provider}"
            
        # Check API keys if needed
        if research_provider in ["OpenAI", "Anthropic"]:
            if not research_api_key:
                return False, f"API key required for {research_provider} research model"
            if len(research_api_key.strip()) < 20:  # Basic key length validation
                return False, f"Invalid {research_provider} research API key"
                
        if synthesis_provider in ["OpenAI", "Anthropic"]:
            if not synthesis_api_key:
                return False, f"API key required for {synthesis_provider} synthesis model"
            if len(synthesis_api_key.strip()) < 20:  # Basic key length validation
                return False, f"Invalid {synthesis_provider} synthesis API key"
                
        return True, "" 