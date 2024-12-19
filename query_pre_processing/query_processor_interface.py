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

# query_processor_interface.py
from abc import ABC, abstractmethod

class QueryProcessor(ABC):
    """
    Abstract class for query preprocessing and augmentation.
    """

    @abstractmethod
    def process(self, query: str) -> str:
        """
        Process or augment the query.

        Args:
            query (str): The original user query.

        Returns:
            str: The processed or augmented query.
        """
        pass
