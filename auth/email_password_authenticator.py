import streamlit as st
from datetime import datetime
from typing import Optional, Dict, Any
from auth.auth_provider_interface import AuthenticationProvider
from auth.user_session import UserSession
from auth.session_manager import SessionManager
import hashlib
import hmac
import secrets
import re
import sqlite3

class EmailPasswordAuthenticator(AuthenticationProvider):
    """Email/Password authentication implementation"""
    
    def __init__(self, db_connection_string):
        """Initialize authenticator with database connection string
        
        Args:
            db_connection_string: Path or connection string to SQLite database
        """
        self.db_connection_string = db_connection_string
        self.session_manager = SessionManager()
        
        # Initialize database schema if needed
        self._init_database()

    def _init_database(self):
        """Initialize database with schema if tables don't exist"""
        try:
            with self._get_db_connection() as conn:
                with open('auth_schema.sql', 'r') as schema_file:
                    conn.executescript(schema_file.read())
                conn.commit()
        except Exception as e:
            print(f"Error initializing database: {e}")
            raise
    
    def _get_db_connection(self):
        """Create a new database connection for the current thread"""
        return sqlite3.connect(self.db_connection_string)
    
    def authenticate(self) -> Optional[UserSession]:
        """Show login form and handle authentication"""
        # Add tabs for login/signup
        login_tab, signup_tab = st.tabs(["Login", "Sign Up"])
        
        # Handle authentication in tabs
        session = None
        with login_tab:
            session = self._show_login_form()
        with signup_tab:
            if not session:  # Only try signup if login didn't succeed
                session = self._show_signup_form()
                
        return session
    
    def _show_login_form(self) -> Optional[UserSession]:
        """Show login form"""
        with st.form("email_password_login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if self._verify_credentials(username, password):
                    session = UserSession(
                        username=username,
                        authenticated=True,
                        auth_provider="email",
                        auth_time=datetime.now(),
                        user_data=self._get_user_data(username)
                    )
                    self.session_manager.set_session(session)
                    st.rerun()
                    #return session
                st.error("Invalid credentials")
        return None
    
    def _show_signup_form(self) -> Optional[UserSession]:
        """Show signup form"""
        with st.form("email_password_signup"):
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            submitted = st.form_submit_button("Sign Up")
            
            if submitted:
                # Validate inputs
                if not all([username, email, password, confirm_password, first_name, last_name]):
                    st.error("All fields are required")
                    return None
                
                if password != confirm_password:
                    st.error("Passwords do not match")
                    return None
                
                if not self._is_valid_email(email):
                    st.error("Invalid email address")
                    return None
                
                # Check if username or email already exists
                if self._user_exists(username, email):
                    st.error("Username or email already exists")
                    return None
                
                # Validate password strength
                is_valid, message = self._validate_password_strength(password)
                if not is_valid:
                    st.error(message)
                    return None
                
                # Create new user
                if self.create_user(
                    username=username,
                    password=password,
                    email=email,
                    first_name=first_name,
                    last_name=last_name
                ):
                    st.success("Account created successfully! Please log in.")
                    # Optionally, automatically log in the user
                    session = UserSession(
                        username=username,
                        authenticated=True,
                        auth_provider="email",
                        auth_time=datetime.now(),
                        user_data=self._get_user_data(username)
                    )
                    self.session_manager.set_session(session)
                    return session
                else:
                    st.error("Error creating account. Please try again.")
        
        return None
    
    def _is_valid_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
        return bool(re.match(pattern, email))
    
    def _user_exists(self, username: str, email: str) -> bool:
        """Check if username or email already exists"""
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT COUNT(*) 
                    FROM users 
                    WHERE username = ? OR email = ?
                ''', (username, email))
                count = cursor.fetchone()[0]
                return count > 0
        except Exception as e:
            print(f"Error checking user existence: {e}")
            return True  # Err on the side of caution
    
    def logout(self) -> None:
        self.session_manager.clear_session()
        st.rerun()
    
    def is_authenticated(self) -> bool:
        session = self.session_manager.get_session()
        return bool(session and session.authenticated)
    
    def _verify_credentials(self, username: str, password: str) -> bool:
        """Verify user credentials against database.
        
        Args:
            username (str): The username to verify
            password (str): The password to verify
        
        Returns:
            bool: True if credentials are valid, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT password_hash, salt 
                    FROM users 
                    WHERE username = ? AND is_active = TRUE
                ''', (username,))
                
                result = cursor.fetchone()
                if not result:
                    return False
                
                stored_hash, salt = result
                
                # Hash the provided password with the stored salt
                password_hash = hashlib.pbkdf2_hmac(
                    'sha256',
                    password.encode('utf-8'),
                    salt.encode('utf-8'),
                    100000  # Number of iterations
                ).hex()
                
                # Compare hashes using constant-time comparison
                return hmac.compare_digest(password_hash, stored_hash)
                
        except Exception as e:
            print(f"Error verifying credentials: {e}")
            return False
    
    def _get_user_data(self, username: str) -> Dict[str, Any]:
        """Retrieve user data from database.
        
        Args:
            username (str): The username to lookup
        
        Returns:
            Dict[str, Any]: User data including profile information
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT 
                        u.email,
                        u.created_at,
                        u.last_login,
                        u.first_name,
                        u.last_name,
                        u.role,
                        u.is_active,
                        COALESCE(p.preferences, '{}') as preferences
                    FROM users u
                    LEFT JOIN user_preferences p ON u.username = p.username
                    WHERE u.username = ?
                ''', (username,))
                
                result = cursor.fetchone()
                if not result:
                    return {}
                
                # Update last login time
                cursor.execute('''
                    UPDATE users 
                    SET last_login = ? 
                    WHERE username = ?
                ''', (datetime.now(), username))
                
                conn.commit()
                
                # Return structured user data
                return {
                    'email': result[0],
                    'created_at': result[1],
                    'last_login': result[2],
                    'first_name': result[3],
                    'last_name': result[4],
                    'role': result[5],
                    'is_active': result[6],
                    'preferences': result[7],
                    'username': username
                }
                
        except Exception as e:
            print(f"Error retrieving user data: {e}")
            return {}
    
    # Helper method for creating new users
    def create_user(self, username: str, password: str, email: str, 
                    first_name: str, last_name: str, role: str = 'user') -> bool:
        """Create a new user with securely hashed password.
        
        Args:
            username (str): Unique username
            password (str): Plain text password to hash
            email (str): User's email
            first_name (str): User's first name
            last_name (str): User's last name
            role (str, optional): User's role. Defaults to 'user'.
        
        Returns:
            bool: True if user was created successfully, False otherwise
        """
        try:
            with self._get_db_connection() as conn:
                cursor = conn.cursor()
                # Generate a random salt
                salt = secrets.token_hex(16)
                
                # Hash the password
                password_hash = hashlib.pbkdf2_hmac(
                    'sha256',
                    password.encode('utf-8'),
                    salt.encode('utf-8'),
                    100000
                ).hex()
                
                # Create user record
                cursor.execute('''
                    INSERT INTO users (
                        username, password_hash, salt, email, 
                        first_name, last_name, role, created_at, 
                        is_active
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    username, password_hash, salt, email,
                    first_name, last_name, role, datetime.now(),
                    True
                ))
                
                # Create default preferences
                cursor.execute('''
                    INSERT INTO user_preferences (username, preferences)
                    VALUES (?, '{}')
                ''', (username,))
                
                conn.commit()
                return True
                
        except Exception as e:
            print(f"Error creating user: {e}")
            return False
    
    def _validate_password_strength(self, password: str) -> tuple[bool, str]:
        """
        Validate password strength.
        Returns (is_valid, message)
        """
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if not any(c.isupper() for c in password):
            return False, "Password must contain at least one uppercase letter"
        
        if not any(c.islower() for c in password):
            return False, "Password must contain at least one lowercase letter"
        
        if not any(c.isdigit() for c in password):
            return False, "Password must contain at least one number"
        
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password):
            return False, "Password must contain at least one special character"
        
        return True, "Password is strong"