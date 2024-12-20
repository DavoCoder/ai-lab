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

# external_knowledge_base_interface.py
from abc import ABC, abstractmethod

class ExternalKnowledgeBase(ABC):
    @abstractmethod
    def connect(self):
        """
        Establish a connection to the external knowledge base.
        """

    @abstractmethod
    def fetch_documents(self):
        """
        Fetch or query documents from the external knowledge base.
        Returns:
            List of documents.
        """

    @abstractmethod
    def detect_updates(self):
        """
        Detect new or updated documents in the external knowledge base.
        Returns:
            List of new or updated documents.
        """

