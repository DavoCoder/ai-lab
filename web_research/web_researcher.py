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

# web_researcher.py
import logging
from typing import List, Tuple
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from anthropic import Anthropic

# Set up logger
logger = logging.getLogger(__name__)

class WebResearcher:
    def __init__(self):
        logger.debug("Initializing WebResearcher")

    @staticmethod
    def perform_research(query: str, urls: List[str], model_provider: str, 
                         model_id: str, depth: int, api_key: str = None) -> list:
        """Performs web research based on the query and URLs."""
        logger.info("Performing research with depth %d", depth)
        sources = []
        
        try:
            if not urls:
                logger.warning("No URLs provided")
                return []
            
            # Ensure urls is a list of strings
            if isinstance(urls, str):
                urls = [urls]

            # Process each URL
            logger.info("Extracting content from %d sources...", len(urls))
            for url in urls:
                if not isinstance(url, str):
                    logger.warning("Skipping invalid URL type: %s", type(url))
                    continue
                try:
                    # Fetch and parse content
                    logger.info("Processing URL: %s", url)
                    content = WebResearcher._fetch_and_parse_url(url)
                    
                    if not content:
                        logger.warning("No content extracted from URL: %s", url)
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
                        logger.debug("Content relevant (score: %.2f) for URL: %s", relevance_score, url)
                        sources.append({
                            "url": url,
                            "title": WebResearcher._extract_title(content),
                            "content": filtered_content,
                            "relevance_score": relevance_score
                        })
                    else:
                        logger.debug("Content not relevant (score: %.2f) for URL: %s", relevance_score, url)
                        
                except WebResearcherException as e:
                    logger.error("Error processing %s: %s", url, str(e), exc_info=True)
                    continue
                    
            # Sort sources by relevance
            sources.sort(key=lambda x: x['relevance_score'], reverse=True)
            logger.info("Successfully processed %d relevant sources", len(sources))
            return sources
            
        except WebResearcherException as e:
            logger.error("Error during research: %s", str(e), exc_info=True)
            raise WebResearcherException(f"Error during research: {str(e)}") from e

    @staticmethod
    def _fetch_and_parse_url(url: str) -> str:
        """Fetches and parses content from a URL."""
        logger.debug("Fetching content from URL: %s", url)
        try:
            # Add headers to mimic browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                             'AppleWebKit/537.36 (KHTML, like Gecko) '
                             'Chrome/91.0.4472.124 Safari/537.36'
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
                logger.debug("Successfully extracted content from URL: %s", url)
                return text
            
            logger.warning("No main content found for URL: %s", url)
            raise WebResearcherException(f"No main content found for URL: {url}")               
            
        except requests.RequestException as e:
            logger.error("Error fetching %s: %s", url, str(e), exc_info=True)
            raise WebResearcherException(f"Error fetching {url}: {str(e)}") from e
        except Exception as e:
            logger.error("Error processing %s: %s", url, str(e), exc_info=True)
            raise WebResearcherException(f"Error processing {url}: {str(e)}") from e

    @staticmethod
    def _extract_title(content: str) -> str:
        """Extracts title from content."""
        title = "Untitled Document"  # Default value
        
        if not content:
            logger.warning("Empty content provided")
        else:
            try:
                soup = BeautifulSoup(content, 'html.parser')
                if soup.find('title'):
                    title = soup.find('title').text.strip()
                elif soup.find('h1'):
                    title = soup.find('h1').text.strip()
                else:
                    logger.warning("No title found in content, using default")
                    
            except ValueError as e:
                logger.error("Invalid content format: %s", str(e), exc_info=True)
            except AttributeError as e:
                logger.error("Error accessing HTML elements: %s", str(e), exc_info=True)
            except Exception as e:
                logger.error("Unexpected error extracting title: %s", str(e), exc_info=True)
        
        return title

    @staticmethod
    def _evaluate_content(content: str, query: str, 
                         model_provider: str, model_id: str, api_key: str = None) -> Tuple[float, str]:
        """Evaluates content relevance and filters it using LLM."""
        logger.debug("Evaluating content relevance using %s model: %s", model_provider, model_id)
        try:
            # Prepare prompt for content evaluation
            prompt = WebResearcher._build_content_evaluation_prompt(content, query)
            
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
                    temperature=0.3, #TODO: Make this a parameter
                    max_tokens=500 #TODO: Make this a parameter
                )
                result = response.content[0].text
                
            else:
                logger.error("Unsupported model provider: %s", model_provider)
                raise WebResearcherException(f"Unsupported model provider: {model_provider}")
                
            # Parse response
            try:
                score_line = [line for line in result.split('\n') if line.startswith('RELEVANCE_SCORE:')][0]
                content_line = [line for line in result.split('\n') if line.startswith('RELEVANT_CONTENT:')][0]
                
                relevance_score = float(score_line.split(':')[1].strip())
                filtered_content = content_line.split(':')[1].strip()
                
                logger.debug("Content evaluation complete. Relevance score: %.2f", relevance_score)
                return relevance_score, filtered_content
                
            except Exception as e:
                logger.error("Error parsing LLM response: %s", str(e), exc_info=True)
                raise WebResearcherException(f"Error parsing LLM response: {str(e)}") from e
                
        except Exception as e:
            logger.error("Error evaluating content: %s", str(e), exc_info=True)
            raise WebResearcherException(f"Error evaluating content: {str(e)}") from e

    @staticmethod
    def synthesize_research(sources: list, query: str, include_citations: bool,
                           model_provider: str, model_id: str, api_key: str = None) -> str:
        """Synthesizes the research findings into a coherent summary."""
        logger.info("Synthesizing research findings from %d sources", len(sources))
        try:
            if not sources:
                logger.warning("No sources available for synthesis")
                return "No relevant sources found."
                
            # Prepare content for synthesis
            source_texts = []
            for idx, source in enumerate(sources, 1):
                source_text = f"Source {idx}: {source['content']}"
                source_texts.append(source_text)
                
            combined_sources = "\n\n".join(source_texts)
            
            # Prepare synthesis prompt
            prompt = WebResearcher._build_synthesis_prompt(query, combined_sources, include_citations)
            
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
                    temperature=0.3, #TODO: Make this a parameter
                    max_tokens=500 #TODO: Make this a parameter
                )
                synthesis = response.content[0].text
                
            else:
                logger.error("Unsupported model provider: %s", model_provider)
                raise WebResearcherException(f"Unsupported model provider: {model_provider}")
                
            logger.info("Successfully synthesized research findings")
            return synthesis
            
        except Exception as e:
            logger.error("Error synthesizing research: %s", str(e), exc_info=True)
            raise WebResearcherException(f"Error synthesizing research: {str(e)}") from e

    @staticmethod
    def _build_synthesis_prompt(query, combined_sources, include_citations):
        return f"""
            Research Query: {query}
            
            Sources:
            {combined_sources}
            
            Please synthesize these sources into a comprehensive research summary.
            {"Include source citations [1], [2], etc. where appropriate." if include_citations else ""}
            Focus on key findings and insights relevant to the research query.
            """
    
    @staticmethod
    def _build_content_evaluation_prompt(content, query):
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
    
class WebResearcherException(Exception):
    """Base exception for web researcher related errors"""
    def __init__(self, message="Web researcher error occurred"):
        self.message = message
        super().__init__(self.message)
