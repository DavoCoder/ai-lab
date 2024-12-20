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

# home.py
from pathlib import Path
import streamlit as st
from ui.labs.app_mode import AppMode

class Home(AppMode):
    """Home page"""
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
