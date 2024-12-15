import streamlit as st
from config import Config
from ui.auth.authenticator_manager import AuthenticationManager
from ui.auth.email_password_authenticator import EmailPasswordAuthenticator
from ui.auth.google_authenticator import GoogleAuthenticator
from ui.labs.mode_config import MODE_INFO, MODE_CLASSES

def setup_authentication():
    auth_manager = AuthenticationManager()
    
    # Register providers
    email_auth = EmailPasswordAuthenticator('users.db')
    google_auth = GoogleAuthenticator(
        client_id=Config.GOOGLE_CLIENT_ID,
        client_secret=Config.GOOGLE_CLIENT_SECRET
    )
    
    auth_manager.register_provider("email", email_auth) 
    #auth_manager.register_provider("Google", google_auth)
    
    return auth_manager


if 'auth_manager' not in st.session_state:
    st.session_state.auth_manager = setup_authentication()
    
auth_manager = st.session_state.auth_manager

# Check authentication and show appropriate content
if not auth_manager.is_authenticated():
    auth_manager.login_form()
else:
    # Show the main application
    st.sidebar.title("AI Playground")
    
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

    # Common Settings (visible in all modes)
    with st.sidebar.expander("Debug Options"):
        show_debug = st.checkbox("Show Debug Info", value=False)
        log_level = st.select_slider(
            "Log Level",
            options=["ERROR", "WARNING", "INFO", "DEBUG"]
        )

    # Help & Documentation
    with st.sidebar.expander("Help & Documentation"):
        st.markdown("""
        - [Documentation](link)
        - [Examples](link)
        - [GitHub Repo](link)
        """)

    # System Status
    st.sidebar.markdown("---")
    st.sidebar.caption("System Status: 🟢 Online")