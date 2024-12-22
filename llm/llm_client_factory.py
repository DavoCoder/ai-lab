from abc import ABC, abstractmethod
from openai import OpenAI
from anthropic import Anthropic
from langchain_core.prompts import ChatPromptTemplate

class LLMClient(ABC):
    def __init__(self):
        """Initialize LLM client with required attributes.
        
        Args:
            api_key (str): API key for the LLM service
        """
        self.client = None  # Will be set by child classes
        self.model_id = None  # Will be set by factory

    @abstractmethod
    def get_completion(self, prompt: str, temperature: float = 0.3, max_tokens: int = 500) -> str:
        pass

class OpenAIClient(LLMClient):
    def __init__(self, api_key: str):
        super().__init__()
        self.client = OpenAI(api_key=api_key)
        
    def get_completion(self, prompt: str, temperature: float = 0.3, max_tokens: int = 500) -> str:
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature
        )
        return response.choices[0].message.content

class AnthropicClient(LLMClient):
    def __init__(self, api_key: str):
        super().__init__()
        self.client = Anthropic(api_key=api_key)
        
    def get_completion(self, prompt: str, temperature: float = 0.3, max_tokens: int = 500) -> str:
        response = self.client.messages.create(
            model=self.model_id,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens
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
