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

# nlp_processor.py
import json
from typing import Dict, Any
from openai import OpenAI
from anthropic import Anthropic
from config import Config

class NLPProcessor:
    """
    NLPProcessor class for processing NLP tasks using OpenAI and Anthropic models.
    """
     # Load all configurations at module import time
    try:
        configs = Config.load_all_configs()
        PROVIDER_MODELS = configs["provider_models"]
        TASK_DESCRIPTIONS = configs["task_descriptions"]
        TASK_SETTINGS = configs["task_settings"]
        SYSTEM_PROMPTS = configs["system_prompts"]
    except Exception as e:
        raise Exception(f"Error initializing NLPProcessor configurations: {str(e)}") from e

    @staticmethod
    def process_task(task_type: str, model_provider: str, model: str, 
                        input_text: str, settings: Dict[str, Any], api_key: str = None):
        if not input_text:
            raise ValueError("Please provide input text")

        if model_provider == "OpenAI":
            return NLPProcessor._process_openai(task_type, model, input_text, settings, api_key)
        elif model_provider == "Anthropic":
            return NLPProcessor._process_anthropic(task_type, model, input_text, settings, api_key)
        else:
            raise NotImplementedError(f"Provider {model_provider} not implemented yet")

    @staticmethod
    def _parse_model_response(task_type: str, result: str) -> Any:
        """
        Parse model response based on task type.
        
        Args:
            task_type (str): Type of NLP task
            result (str): Raw response from the model
            
        Returns:
            Any: Parsed result in appropriate format for the task
        """
        if task_type == "Text Classification":
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {
                    "label": "error",
                    "confidence": 0.0,
                    "error": "Failed to parse model response"
                }
                
        elif task_type == "Named Entity Recognition":
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return []
                
        else:  # Text Generation, Summarization, Translation
            return result

    @staticmethod
    def _get_formatted_system_prompt(task_type: str, settings: Dict[str, Any]) -> str:
        """
        Get and format system prompt for the given task type.
        
        Args:
            task_type (str): Type of NLP task
            settings (Dict[str, Any]): Task settings containing any needed format variables
            
        Returns:
            str: Formatted system prompt
            
        Raises:
            KeyError: If task_type is not found in SYSTEM_PROMPTS
        """
        system_prompt = NLPProcessor.SYSTEM_PROMPTS[task_type]
        
        if task_type == "Translation":
            system_prompt = system_prompt.format(
                source_lang=settings.get('source_lang', 'English'),
                target_lang=settings.get('target_lang', 'French')
            )
            
        return system_prompt

    @staticmethod
    def _process_openai(task_type: str, model: str, input_text: str, 
                        settings: Dict[str, Any], api_key: str) -> Any:
        client = OpenAI(api_key=api_key)
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": NLPProcessor._get_formatted_system_prompt(task_type, settings)},
                    {"role": "user", "content": input_text}
                ],
                temperature=settings.get("temperature", 0.7),
                max_tokens=settings.get("max_tokens", 500),
                top_p=settings.get("top_p", 1.0),
                frequency_penalty=settings.get("frequency_penalty", 0.0),
                presence_penalty=settings.get("presence_penalty", 0.0)
            )
            
            result = response.choices[0].message.content
            return NLPProcessor._parse_model_response(task_type, result)

        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}") from e

    @staticmethod
    def _process_anthropic(task_type: str, model: str, input_text: str, 
                            settings: Dict[str, Any], api_key: str) -> Any:
        client = Anthropic(api_key=api_key)
        
        try:
            response = client.messages.create(
                model=model,
                max_tokens=settings.get("max_tokens", 500),
                temperature=settings.get("temperature", 0.7),
                top_p=settings.get("top_p", 1.0),
                messages=[{
                    "role": "user",
                    "content": f"{NLPProcessor._get_formatted_system_prompt(task_type, settings)}\n\n{input_text}"
                }]
            )
            
            result = response.content[0].text
            return NLPProcessor._parse_model_response(task_type, result)

        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}") from e
