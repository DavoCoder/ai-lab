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

# agent_lab.py

import re
from typing import List
import streamlit as st
from ui.labs.app_mode import AppMode
from llm.llm_client_factory import LLMClientFactory, LLMClientFactoryException
from llm.prompt_builder import PromptBuilder

class AgentLab(AppMode):
    """Agent Lab for demonstrating different agentic workflows"""

    TASK_CONFIGS = {
        "Code Analysis": {
            "input_type": "text_input",
            "label": "Analysis Focus",
            "placeholder": "e.g., 'Identify potential performance issues' or 'Explain the main logic'"
        },
        "Code Generation": {
            "input_type": "text_input",
            "label": "Generation Requirements",
            "placeholder": "Describe the code you want to generate"
        },
        "Code Documentation": {
            "input_type": "select",
            "label": "Documentation Style",
            "options": ["Google Style", "NumPy Style", "RST Style"],
            "instruction_template": "Generate {} documentation for this code"
        },
        "Code Review": {
            "input_type": "multiselect",
            "label": "Review Focus Areas",
            "options": ["Best Practices", "Security", "Performance", "Readability", "Error Handling"],
            "instruction_template": "Review code focusing on: {}"
        }
    }

    def __init__(self):
        self.llm_provider = None
        self.llm_model = None
        self.llm_api_key = None
        
    @staticmethod
    def render():
        st.title("🤖 Agent Lab")
        
        # Sidebar configuration
        with st.sidebar:
            st.subheader("Agent Configuration")
            
            # LLM Settings
            AgentLab.llm_provider = st.selectbox(
                "Select LLM Provider", ["OpenAI", "Anthropic"])
            
            if AgentLab.llm_provider == "OpenAI":
                AgentLab.llm_model = st.selectbox(
                    "Select LLM Model", ["gpt-4", "gpt-3.5-turbo"])
            else:
                AgentLab.llm_model = st.selectbox(
                    "Select LLM Model", 
                    ["claude-3-opus-20240229", "claude-3-sonnet-20240229"])
                
            AgentLab.llm_api_key = st.text_input(
                f"{AgentLab.llm_provider} API Key", type="password")

        # Main content area
        tab1, tab2, tab3 = st.tabs([
            "Code Assistant Agent",
            "Agent 2", 
            "Agent 3"
        ])

        with tab1:
            AgentLab._render_code_agent()
            
        with tab2:
            st.success("Not implemented yet!")
            
        with tab3:
            st.success("Not implemented yet!")

    @staticmethod
    def _render_code_agent():
        st.header("Code Assistant Agent")
        st.markdown("""
        This agent helps with code-related tasks using:
        1. Code analysis and understanding
        2. Documentation lookup
        3. Code generation and explanation
        """)
        
        # Task selection
        task_type = st.selectbox(
            "Select Task Type",
            ["Code Analysis", "Code Generation", "Code Documentation", "Code Review"]
        )
        
        # Code input area with language selection
        programming_language = st.selectbox(
            "Programming Language",
            ["Python", "JavaScript", "Java", "C++", "Other"]
        )
        
        code_input = st.text_area(
            "Enter Code",
            height=200,
            help="Paste your code here or describe what you want to generate"
        )
        
        # Task-specific inputs
        instruction = AgentLab._get_task_instruction(task_type)
        
        if st.button("Execute Code Agent") and code_input and instruction:
            try:
                with st.spinner("Processing code..."):
                    # Initialize LLM client
                    llm_client = LLMClientFactory.create_client(
                        provider=AgentLab.llm_provider,
                        model_id=AgentLab.llm_model,
                        api_key=AgentLab.llm_api_key
                    )
                    
                    # Prepare the prompt based on task type
                    system_prompt = PromptBuilder.build_code_agent_system_prompt(
                        task_type=task_type,
                        language=programming_language
                    )
                    
                    # Prepare the user message
                    user_message = PromptBuilder.build_code_agent_user_prompt(
                        instruction=instruction,
                        programming_language=programming_language,
                        code_input=code_input,
                        task_type=task_type
                    )

                    response = llm_client.get_completion(user_prompt=user_message, 
                                                system_prompt=system_prompt)
                    
                    # Display results
                    st.success("Code processing completed!")
                    
                    # Format and display the response
                    st.markdown("### Results")
                    if task_type == "Code Generation":
                        # Extract code blocks from response
                        code_blocks = AgentLab._extract_code_blocks(response)
                        if code_blocks:
                            for i, code in enumerate(code_blocks, 1):
                                st.code(code, language=programming_language.lower())
                                # Add copy button for each code block
                                st.button(f"📋 Copy Code Block {i}", 
                                        key=f"copy_btn_{i}", 
                                        on_click=lambda c=code: st.write(c))
                        
                    elif task_type == "Code Analysis":
                        st.markdown(response)
                        
                    elif task_type == "Code Documentation":
                        st.markdown("#### Generated Documentation")
                        st.markdown(response)
                        
                    else:  # Code Review
                        st.markdown("#### Review Comments")
                        st.markdown(response)
                    
            except LLMClientFactoryException as e:
                st.error(f"Error processing code: {str(e)}")
        else:
            st.warning("Please fill in all fields before executing the agent.")

    @staticmethod
    def _extract_code_blocks(text: str) -> List[str]:
        """Extract code blocks from markdown-formatted text."""
        
        # Pattern to match code blocks with or without language specification
        pattern = r"```(?:\w+)?\n(.*?)```"
        matches = re.finditer(pattern, text, re.DOTALL)
        
        return [match.group(1).strip() for match in matches]

    @staticmethod
    def _get_task_instruction(task_type: str) -> str:
        """Get instruction based on task type configuration."""
        config = AgentLab.TASK_CONFIGS[task_type]
        instruction = ""
        
        if config["input_type"] == "text_input":
            instruction = st.text_input(
                config["label"],
                placeholder=config["placeholder"]
            )
        elif config["input_type"] == "select":
            style = st.selectbox(config["label"], config["options"])
            instruction = config["instruction_template"].format(style)
        elif config["input_type"] == "multiselect":
            focus = st.multiselect(config["label"], config["options"])
            instruction = config["instruction_template"].format(", ".join(focus))
            
        return instruction
