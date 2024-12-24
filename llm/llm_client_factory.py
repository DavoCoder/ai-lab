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
from typing import Dict, Any
from openai import OpenAI
from anthropic import Anthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

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

class LLMClientFactory:
    def __init__(self):
        raise TypeError("LLMClientFactory cannot be instantiated")
    
    @staticmethod
    def create_client(provider: str, api_key: str, model_id: str) -> LLMClient:
        if provider == "OpenAI":
            client = OpenAIClient(api_key)
        elif provider == "Anthropic":
            client = AnthropicClient(api_key)
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
        else:
            raise LLMClientFactoryException(f"Unsupported model provider: {provider}")
        
        return client
    
class LLMClientFactoryException(Exception):
    """Base exception for LLM client factory related errors"""
    def __init__(self, message="LLM client factory error occurred"):
        self.message = message
        super().__init__(self.message)

class PromptBuilder():

    @staticmethod
    def build_qa_chain_prompt():
        system_prompt = (
            "Use the given context to answer the question. "
            "If you don't know the answer, say you don't know. "
            "Use three sentence maximum and keep the answer concise. "
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{input}"),
            ]
        )
        return prompt
    
    @staticmethod
    def build_synthesis_prompt(query, combined_sources, include_citations):
        return f"""
            Research Query: {query}
            
            Sources:
            {combined_sources}
            
            Please synthesize these sources into a comprehensive research summary.
            {"Include source citations [1], [2], etc. where appropriate." if include_citations else ""}
            Focus on key findings and insights relevant to the research query.
            """

    @staticmethod
    def build_content_evaluation_prompt(content, query):
        return f"""
            Research Query: {query}
            
            Content to evaluate:
            {content[:2000]}  # Limit content length for API
            
            Please perform two tasks:
            1. Rate the relevance of this content to the research query on a scale of 0.0 to 1.0
            2. Extract and summarize the most relevant information from this content
            
            Format your response as:
            RELEVANCE_SCORE: [score]
            RELEVANT_CONTENT: [extracted content]
            """

    @staticmethod
    def build_code_agent_system_prompt(task_type: str, language: str) -> str:
        """Build the system prompt based on task type and programming language."""
        
        base_prompt = f"You are an expert {language} developer and code assistant. "
        
        task_prompts = {
            "Code Analysis": """
                Analyze the provided code and provide insights about its:
                1. Main functionality and logic flow
                2. Potential issues or improvements
                3. Code complexity and performance considerations
                Be specific and provide examples where relevant.
            """,
            
            "Code Generation": f"""
                Generate clean, well-documented {language} code based on the requirements.
                Include:
                1. Error handling
                2. Input validation
                3. Comments explaining key logic
                4. Best practices for {language}
                Return only the code block without additional explanation unless specifically requested.
            """,
            
            "Code Documentation": f"""
                Generate comprehensive documentation for the provided {language} code.
                Include:
                1. Function/class purpose
                2. Parameter descriptions
                3. Return value details
                4. Usage examples
                5. Any important notes or warnings
                Follow the specified documentation style guide.
            """,
            
            "Code Review": f"""
                Perform a thorough code review focusing on:
                1. Code quality and best practices
                2. Potential bugs or issues
                3. Security considerations
                4. Performance optimization opportunities
                5. {language}-specific improvements
                Provide specific recommendations for each issue found.
            """
        }
        
        return base_prompt + task_prompts.get(task_type, "")
    
    @staticmethod
    def build_code_agent_user_prompt(
        instruction: str,
        programming_language: str,
        code_input: str,
        task_type: str = None,
        additional_context: Dict[str, Any] = None
    ) -> str:
        """
        Build the user prompt for the code agent.

        Args:
            instruction (str): The main instruction or task description
            programming_language (str): The programming language being used
            code_input (str): The code to be processed
            task_type (str, optional): Type of task being performed
            additional_context (dict, optional): Any additional context needed

        Returns:
            str: Formatted user prompt
        """
        # Start with the basic prompt structure
        prompt_parts = [
            f"Task: {instruction}",
            f"Programming Language: {programming_language}",
        ]

        # Add task-specific context if provided
        if task_type:
            prompt_parts.append(f"Task Type: {task_type}")

        # Add any additional context
        if additional_context:
            for key, value in additional_context.items():
                prompt_parts.append(f"{key}: {value}")

        # Add the code block
        prompt_parts.extend([
            "Code:",
            f"```{programming_language.lower()}",
            code_input,
            "```"
        ])

        # Join all parts with newlines
        return "\n".join(prompt_parts)
