from abc import ABC, abstractmethod
from typing import Optional
from ui.auth.user_session import UserSession

class AuthenticationProvider(ABC):
    """Abstract base class for authentication providers"""
    
    @abstractmethod
    def authenticate(self) -> Optional[UserSession]:
        """Authenticate user and return session data if successful"""
        pass
    
    @abstractmethod
    def logout(self) -> None:
        """Handle logout process"""
        pass
    
    @abstractmethod
    def is_authenticated(self) -> bool:
        """Check if current session is authenticated"""
        pass