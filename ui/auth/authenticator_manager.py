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

# authenticator_manager.py
from typing import Dict
import streamlit as st
from ui.auth.auth_provider_interface import AuthenticationProvider
from ui.auth.session_manager import SessionManager

class AuthenticationManager:
    """Manages multiple authentication providers"""
    
    def __init__(self):
        self.session_manager = SessionManager()
        self.session_manager.init_session()
        self.providers: Dict[str, AuthenticationProvider] = {}
    
    def register_provider(self, name: str, provider: AuthenticationProvider):
        """Register a new authentication provider"""
        # Convert name to lowercase for consistent lookup
        self.providers[name.lower()] = provider
    
    def login_form(self):
        """Display login form with all available providers"""
        st.markdown("## 🔐 Login")
        
        provider_names = list(self.providers.keys())
        selected_provider = st.radio("Login method", provider_names)
        
        provider = self.providers[selected_provider.lower()]
        return provider.authenticate()
    
    def logout(self):
        """Logout from current session"""
        if st.sidebar.button("Logout"):
            current_session = self.session_manager.get_session()
            if current_session:
                # Convert provider name to lowercase for lookup
                provider = self.providers[current_session.auth_provider.lower()]
                provider.logout()
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated with any provider"""
        return bool(self.session_manager.get_session())
