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
from llm.llm_client_factory import LLMClientFactory, LLMClientFactoryException
from llm.prompt_builder import PromptBuilder

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
                    
            except (ValueError, AttributeError) as e:
                logger.error("Error extracting title: %s", str(e), exc_info=True)
        
        return title

    @staticmethod
    def _evaluate_content(content: str, query: str, 
                         model_provider: str, model_id: str, api_key: str = None) -> Tuple[float, str]:
        """Evaluates content relevance and filters it using LLM."""
        logger.debug("Evaluating content relevance using %s model: %s", model_provider, model_id)
        try:
            # Prepare prompt for content evaluation
            prompt = PromptBuilder.build_content_evaluation_prompt(content, query)
            
            # Get LLM client and completion
            client = LLMClientFactory.create_client(model_provider, api_key, model_id)
            #TODO Evaluate if adding temperature and max tokens makes sense
            result = client.get_completion(user_prompt=prompt, system_prompt=None)
            
            # Parse response
            try:
                score_line = [line for line in result.split('\n') if line.startswith('RELEVANCE_SCORE:')][0]
                content_line = [line for line in result.split('\n') if line.startswith('RELEVANT_CONTENT:')][0]
                
                relevance_score = float(score_line.split(':')[1].strip())
                filtered_content = content_line.split(':')[1].strip()
                
                logger.debug("Content evaluation complete. Relevance score: %.2f", relevance_score)
                return relevance_score, filtered_content
                
            except LLMClientFactoryException as e:
                logger.error("Error creating LLM client: %s", str(e), exc_info=True)
                raise WebResearcherException(f"Error creating LLM client: {str(e)}") from e
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
            source_texts = [f"Source {idx}: {source['content']}" 
                          for idx, source in enumerate(sources, 1)]
            combined_sources = "\n\n".join(source_texts)
            
            # Prepare synthesis prompt
            prompt = PromptBuilder.build_synthesis_prompt(query, combined_sources, include_citations)
            
            # Get LLM client and completion
            client = LLMClientFactory.create_client(model_provider, api_key, model_id)
            #TODO Evaluate if adding temperature and max tokens makes sense
            synthesis = client.get_completion(user_prompt=prompt, system_prompt=None)
            
            logger.info("Successfully synthesized research findings")
            return synthesis
        
        except LLMClientFactoryException as e:
            logger.error("Error creating LLM client: %s", str(e), exc_info=True)
            raise WebResearcherException(f"Error creating LLM client: {str(e)}") from e
        except Exception as e:
            logger.error("Error synthesizing research: %s", str(e), exc_info=True)
            raise WebResearcherException(f"Error synthesizing research: {str(e)}") from e
    
class WebResearcherException(Exception):
    """Base exception for web researcher related errors"""
    def __init__(self, message="Web researcher error occurred"):
        self.message = message
        super().__init__(self.message)
