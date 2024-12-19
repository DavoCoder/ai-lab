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

# auth_provider_interface.py
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