from typing import Dict, Any, List, Tuple
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from anthropic import Anthropic

class WebResearcher:
    def __init__(self):
        pass

    def search(self, query: str, urls: List[str], model_provider: str, model_id: str, 
               depth: int = 2, include_citations: bool = True, api_key: str = None) -> Dict[str, Any]:
        """
        Perform web research based on query and URLs.
        
        Args:
            query (str): Research query
            urls (List[str]): List of URLs to research
            model_provider (str): LLM provider (OpenAI or Anthropic)
            model_id (str): Specific model ID to use
            depth (int): Search depth (1-5)
            include_citations (bool): Whether to include citations
            
        Returns:
            Dict[str, Any]: Research results including synthesis and sources
        """
        try:
            # Process URLs and gather sources
            sources = self.perform_research(
                query=query,
                urls=urls,
                model_provider=model_provider,
                model_id=model_id,
                depth=depth,
                api_key=api_key
            )
            
            # Synthesize findings
            synthesis = self.synthesize_research(
                sources=sources,
                query=query,
                include_citations=include_citations,
                model_provider=model_provider,
                model_id=model_id,
                api_key=api_key
            )
            
            return {
                "synthesis": synthesis,
                "sources": sources
            }
            
        except Exception as e:
            print(f"Error during research: {str(e)}")
            return {"synthesis": "", "sources": []}

    @staticmethod
    def perform_research(query: str, urls: List[str], model_provider: str, 
                         model_id: str, depth: int, api_key: str = None) -> list:
        """
        Performs web research based on the query and URLs.
        
        Args:
            query (str): The research query
            urls (List[str]): List of URLs to process
            model_provider (str): The LLM provider to use
            model_id (str): The specific model ID to use
            depth (int): How deep to search (1-5)
        
        Returns:
            list: List of dictionaries containing source information and content
        """
        sources = []
        
        try:
            if not urls:
                print("No URLs provided.")
                return []
                
            # Process each URL
            print("Extracting content from sources...")
            for url in urls:
                try:
                    # Fetch and parse content
                    content = WebResearcher._fetch_and_parse_url(url)
                    
                    if not content:
                        continue
                    
                    # Evaluate content relevance using LLM
                    relevance_score, filtered_content = WebResearcher._evaluate_content(
                        content=content,
                        query=query,
                        model_provider=model_provider,
                        model_id=model_id,
                        api_key=api_key
                    )
                    
                    if relevance_score > 0.5:  # Threshold for relevance
                        sources.append({
                            "url": url,
                            "title": WebResearcher._extract_title(content),
                            "content": filtered_content,
                            "relevance_score": relevance_score
                        })
                        
                except Exception as e:
                    print(f"Error processing {url}: {str(e)}")
                    continue
                    
            # Sort sources by relevance
            sources.sort(key=lambda x: x['relevance_score'], reverse=True)
            
            return sources
            
        except Exception as e:
            print(f"Error during research: {str(e)}")
            return []

    @staticmethod
    def _fetch_and_parse_url(url: str) -> str:
        """Fetches and parses content from a URL."""
        try:
            # Add headers to mimic browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            # Fetch content
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()
                
            # Extract main content
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            
            if main_content:
                # Clean text
                text = ' '.join(main_content.stripped_strings)
                text = ' '.join(text.split())  # Normalize whitespace
                return text
            return ""
            
        except Exception as e:
            print(f"Error fetching {url}: {str(e)}")
            return ""

    @staticmethod
    def _extract_title(content: str) -> str:
        """Extracts title from content."""
        try:
            soup = BeautifulSoup(content, 'html.parser')
            title = soup.find('title')
            if title:
                return title.text.strip()
            h1 = soup.find('h1')
            if h1:
                return h1.text.strip()
            return "Untitled Document"
        except:
            return "Untitled Document"

    @staticmethod
    def _evaluate_content(content: str, query: str, 
                         model_provider: str, model_id: str, api_key: str = None) -> Tuple[float, str]:
        """
        Evaluates content relevance and filters it using LLM.
        
        Returns:
            tuple: (relevance_score, filtered_content)
        """
        try:
            # Prepare prompt for content evaluation
            prompt = f"""
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
            
            # Initialize appropriate client based on provider
            if model_provider == "OpenAI":
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                result = response.choices[0].message.content
                
            elif model_provider == "Anthropic":
                client = Anthropic(api_key=api_key)
                response = client.messages.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                result = response.content[0].text
                
            else:
                raise ValueError(f"Unsupported model provider: {model_provider}")
                
            # Parse response
            try:
                score_line = [line for line in result.split('\n') if line.startswith('RELEVANCE_SCORE:')][0]
                content_line = [line for line in result.split('\n') if line.startswith('RELEVANT_CONTENT:')][0]
                
                relevance_score = float(score_line.split(':')[1].strip())
                filtered_content = content_line.split(':')[1].strip()
                
                return relevance_score, filtered_content
                
            except:
                return 0.0, ""
                
        except Exception as e:
            print(f"Error evaluating content: {str(e)}")
            return 0.0, ""

    @staticmethod
    def synthesize_research(sources: list, query: str, include_citations: bool,
                           model_provider: str, model_id: str, api_key: str = None) -> str:
        """
        Synthesizes the research findings into a coherent summary.
        
        Args:
            sources (list): List of source dictionaries
            query (str): Original research query
            include_citations (bool): Whether to include citations
            model_provider (str): The LLM provider to use
            model_id (str): The specific model ID to use
            
        Returns:
            str: Synthesized research findings
        """
        try:
            if not sources:
                return "No relevant sources found."
                
            # Prepare content for synthesis
            source_texts = []
            for idx, source in enumerate(sources, 1):
                source_text = f"Source {idx}: {source['content']}"
                source_texts.append(source_text)
                
            combined_sources = "\n\n".join(source_texts)
            
            # Prepare synthesis prompt
            prompt = f"""
            Research Query: {query}
            
            Sources:
            {combined_sources}
            
            Please synthesize these sources into a comprehensive research summary.
            {"Include source citations [1], [2], etc. where appropriate." if include_citations else ""}
            Focus on key findings and insights relevant to the research query.
            """
            
            # Get synthesis from LLM
            if model_provider == "OpenAI":
                client = OpenAI(api_key=api_key)
                response = client.chat.completions.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                synthesis = response.choices[0].message.content
                
            elif model_provider == "Anthropic":
                client = Anthropic(api_key=api_key)
                response = client.messages.create(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3
                )
                synthesis = response.content[0].text
                
            else:
                raise ValueError(f"Unsupported model provider: {model_provider}")
                
            return synthesis
            
        except Exception as e:
            print(f"Error synthesizing research: {str(e)}")
            return "Error synthesizing research findings."