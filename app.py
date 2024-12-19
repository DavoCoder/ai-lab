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

# app.py
import streamlit as st
import logging
from config import Config
from ui.auth.authenticator_manager import AuthenticationManager
from ui.auth.email_password_authenticator import EmailPasswordAuthenticator
from ui.auth.google_authenticator import GoogleAuthenticator
from ui.labs.mode_config import MODE_INFO, MODE_CLASSES

def setup_authentication():
    auth_manager = AuthenticationManager()
    
    # Register providers
    email_auth = EmailPasswordAuthenticator(Config.AUTH_DB_PATH)
    google_auth = GoogleAuthenticator(
        client_id=Config.GOOGLE_CLIENT_ID,
        client_secret=Config.GOOGLE_CLIENT_SECRET
    )
    
    auth_manager.register_provider("email", email_auth) 
    #auth_manager.register_provider("Google", google_auth)
    
    return auth_manager


st.set_page_config(
    page_title="AI Lab",
    page_icon="🧪",
    menu_items={
        'Get Help': 'https://github.com/DavoCoder/ai-lab',
        'Report a bug': "https://github.com/DavoCoder/ai-lab/issues",
        'About': """
        # AI Lab 🧪
        
        An experimental platform for exploring AI concepts and tools.
        
        Version: 1.0.0
        
        Created by: DavoCoder
        
        [GitHub Repository](https://github.com/DavoCoder/ai-lab)
        """
    }
)
if 'auth_manager' not in st.session_state:
    st.session_state.auth_manager = setup_authentication()
    
auth_manager = st.session_state.auth_manager

# Check authentication and show appropriate content
if not auth_manager.is_authenticated():
    auth_manager.login_form()
else:
    # Show the main application
    st.sidebar.title("AI Lab")
    
    # Show logged-in user info and logout button
    session = auth_manager.session_manager.get_session()
    st.sidebar.markdown(f"👤 Logged in as: **{session.username}**")
    auth_manager.logout()

    # Validate and set environment variables at startup
    Config.validate()
    #Config.set_environment()
    
    # Main Mode Selection
    app_mode = st.sidebar.selectbox(
        "Select Mode",
        list(MODE_CLASSES.keys())
    )

    st.title(MODE_INFO[app_mode]["title"])
    st.markdown(MODE_INFO[app_mode]["description"])
    st.markdown("---")

    # Render selected mode
    MODE_CLASSES[app_mode].render()

    # Help & Documentation
    with st.sidebar.expander("Help & Documentation"):
        st.markdown("""
        - [Documentation](https://github.com/DavoCoder/ai-lab)
        - [Examples](https://github.com/DavoCoder/ai-lab)
        - [GitHub Repo](https://github.com/DavoCoder/ai-lab)
        """)



#   Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler('all_logs.log')  # File output
        ]
    )

    # System Status
    st.sidebar.markdown("---")
    st.sidebar.caption("Developed by DavoCoder 🤖 2024")
