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

# google_authenticator.py
from typing import Optional
import streamlit as st
from google_auth_oauthlib.flow import Flow
from ui.auth.auth_provider_interface import AuthenticationProvider
from ui.auth.user_session import UserSession
from ui.auth.session_manager import SessionManager

class GoogleAuthenticator(AuthenticationProvider):
    """Google OAuth authentication implementation"""
    
    def __init__(self, client_id: str, client_secret: str):
        self.client_id = client_id
        self.client_secret = client_secret
        self.session_manager = SessionManager()
    
    def authenticate(self) -> Optional[UserSession]:
        """Handle Google OAuth authentication"""
        flow = Flow.from_client_secrets_file(
            'client_secrets.json',
            scopes=['openid', 'email', 'profile']
        )
        
        authorization_url, state = flow.authorization_url()
        print(f"Authorization URL: {authorization_url}")
        print(f"State: {state}")
        
        if st.button("Login with Google"):
            st.markdown(f"[Login with Google]({authorization_url})")
            # Handle OAuth callback and create session
            # This is simplified - need proper OAuth callback handling
    
    def logout(self) -> None:
        self.session_manager.clear_session()
        st.rerun()
    
    def is_authenticated(self) -> bool:
        session = self.session_manager.get_session()
        return bool(session and session.authenticated)
