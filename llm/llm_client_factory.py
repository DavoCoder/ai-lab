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

# llm_client_factory.py

from abc import ABC, abstractmethod
from openai import OpenAI
from anthropic import Anthropic
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

class LLMClient(ABC):
    def __init__(self):
        """Initialize LLM client with required attributes.
        
        Args:
            api_key (str): API key for the LLM service
        """
        self.client = None  # Will be set by child classes
        self.model_id = None  # Will be set by factory

    @abstractmethod
    def get_completion(self, user_prompt: str, system_prompt: str, 
                       temperature: float = 0.3, max_tokens: int = 500, top_p: float = 1.0,
                       frequency_penalty: float = 0.0, presence_penalty: float = 0.0) -> str:
        pass

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str):
        super().__init__()
        self.client = OpenAI(api_key=api_key)
        
    def get_completion(self, user_prompt: str, system_prompt: str, temperature: float = 0.3, 
                       max_tokens: int = 500, top_p: float = 1.0, frequency_penalty: float = 0.0, 
                       presence_penalty: float = 0.0) -> str:
        messages = [{"role": "user", "content": user_prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
            
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        return response.choices[0].message.content

class AnthropicClient(LLMClient):
    def __init__(self, api_key: str):
        super().__init__()
        self.client = Anthropic(api_key=api_key)
        
    def get_completion(self, user_prompt: str, system_prompt: str, temperature: float = 0.3, 
                       max_tokens: int = 500, top_p: float = 1.0, frequency_penalty: float = 0.0, 
                       presence_penalty: float = 0.0) -> str:
        
        messages = [{"role": "user", "content": user_prompt}]
        if system_prompt:
            messages = [{
                    "role": "user",
                    "content": f"{system_prompt}\n\n{user_prompt}"
                }]
            
        response = self.client.messages.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p
        )
        return response.content[0].text

class GoogleClient(LLMClient):
    def __init__(self, api_key: str):
        super().__init__()
        self.client = ChatGoogleGenerativeAI(
            model="gemini-pro",
            api_key=api_key
        )
        
    def get_completion(self, user_prompt: str, system_prompt: str, temperature: float = 0.3, 
                       max_tokens: int = 500, top_p: float = 1.0, frequency_penalty: float = 0.0, 
                       presence_penalty: float = 0.0) -> str:
        # Combine system prompt and user prompt if both exist
        prompt = f"{system_prompt}\n\n{user_prompt}" if system_prompt else user_prompt
            
        response = self.client.invoke(prompt)
        return response.content
    
class DeepSeekClient(LLMClient):
    def __init__(self, api_key: str):
        super().__init__()
        self.client = OpenAI(base_url="https://api.deepseek.com", api_key=api_key)
        
    def get_completion(self, user_prompt: str, system_prompt: str, temperature: float = 0.3, 
                       max_tokens: int = 500, top_p: float = 1.0, frequency_penalty: float = 0.0, 
                       presence_penalty: float = 0.0) -> str:
        messages = [{"role": "user", "content": user_prompt}]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
            
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty
        )
        return response.choices[0].message.content

class LLMClientFactory:
    def __init__(self):
        raise TypeError("LLMClientFactory cannot be instantiated")
    
    @staticmethod
    def create_client(provider: str, api_key: str, model_id: str) -> LLMClient:
        if provider == "OpenAI":
            client = OpenAIClient(api_key)
        elif provider == "Anthropic":
            client = AnthropicClient(api_key)
        elif provider == "Google":
            client = GoogleClient(api_key)
        elif provider == "DeepSeek":
            client = DeepSeekClient(api_key)
        else:
            raise LLMClientFactoryException(f"Unsupported model provider: {provider}")
        
        client.model_id = model_id
        return client

    @staticmethod
    def create_chat_client(provider: str, model: str, api_key: str):
        if provider == "OpenAI":
            client = ChatOpenAI(
                model_name=model,
                openai_api_key=api_key,
                temperature=0
            )
        elif provider == "Anthropic":
            client = ChatAnthropic(
                model=model,
                anthropic_api_key=api_key,
                temperature=0
            )
        elif provider == "Google":
            client = ChatGoogleGenerativeAI(
                model=model,
                google_api_key=api_key,
                temperature=0
            )
        elif provider == "DeepSeek":
            client = ChatOpenAI(
                model_name=model, 
                openai_api_base="https://api.deepseek.com",
                openai_api_key=api_key,
                temperature=0
            )
        else:
            raise LLMClientFactoryException(f"Unsupported model provider: {provider}")
        
        return client
    
class LLMClientFactoryException(Exception):
    """Base exception for LLM client factory related errors"""
    def __init__(self, message="LLM client factory error occurred"):
        self.message = message
        super().__init__(self.message)
