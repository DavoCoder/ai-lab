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

# session_manager.py
import streamlit as st
from typing import Optional
from ui.auth.user_session import UserSession

class SessionManager:
    """Manages user session state"""
    
    @staticmethod
    def init_session():
        """Initialize session state variables"""
        if 'user_session' not in st.session_state:
            st.session_state.user_session = None
    
    @staticmethod
    def set_session(session: UserSession):
        st.session_state.user_session = session
    
    @staticmethod
    def clear_session():
        st.session_state.user_session = None
    
    @staticmethod
    def get_session() -> Optional[UserSession]:
        return st.session_state.user_session