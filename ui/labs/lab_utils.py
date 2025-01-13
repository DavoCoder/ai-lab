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

# lab_utils.py

from typing import Dict, Any

class LabUtils:
    """
    Lab Utils
    """

    @staticmethod
    def get_slider_params(setting_name: str, setting_config: Dict[str, Any]) -> tuple[str, float, float, float]:
        """Get slider parameters from setting configuration.
        
        Args:
            setting_name: Name of the setting
            setting_config: Configuration dictionary for the setting
            
        Returns:
            Tuple of (display_name, min_value, max_value, default_value)
        """
        return (
            setting_name.replace("_", " ").title(),
            setting_config["min"],
            setting_config["max"],
            setting_config["default"]
        )
