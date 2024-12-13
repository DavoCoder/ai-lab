from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime

@dataclass
class UserSession:
    """Represents user session data"""
    username: str
    authenticated: bool
    auth_provider: str
    auth_time: datetime
    user_data: Dict[str, Any]