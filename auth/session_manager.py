import streamlit as st
from typing import Optional
from auth.user_session import UserSession

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