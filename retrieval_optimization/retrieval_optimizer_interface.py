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

# retrieval_optimizer_interface.py
from abc import ABC, abstractmethod

class RetrievalOptimizer(ABC):
    """
    Abstract class for optimizing the retrieval process.
    """

    @abstractmethod
    def optimize(self, query: str, retrieved_docs: list) -> list:
        """
        Optimize the retrieved documents.

        Args:
            query (str): The search query.
            retrieved_docs (list): A list of retrieved documents.

        Returns:
            list: A list of optimized/re-ranked documents.
        """
