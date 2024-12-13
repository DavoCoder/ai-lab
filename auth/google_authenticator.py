import streamlit as st
from typing import Optional
from google_auth_oauthlib.flow import Flow
from auth.auth_provider_interface import AuthenticationProvider
from auth.user_session import UserSession
from auth.session_manager import SessionManager

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
        
        if st.button("Login with Google"):
            st.markdown(f"[Login with Google]({authorization_url})")
            # Handle OAuth callback and create session
            # This is simplified - you'll need proper OAuth callback handling
            
        return None
    
    def logout(self) -> None:
        self.session_manager.clear_session()
        st.rerun()
    
    def is_authenticated(self) -> bool:
        session = self.session_manager.get_session()
        return bool(session and session.authenticated)