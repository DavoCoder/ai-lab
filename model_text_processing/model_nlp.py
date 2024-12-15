from typing import Dict, Any
from openai import OpenAI
from anthropic import Anthropic


class ModelNLP:
    PROVIDER_MODELS = {
        "OpenAI": {
            "Text Generation": {
                "models": ["gpt-4", "gpt-3.5-turbo"],
                "model_ids": {
                    "gpt-4": "gpt-4-0125-preview",
                    "gpt-3.5-turbo": "gpt-3.5-turbo-0125"
                }
            },
            "Text Classification": {
                "models": ["gpt-4", "gpt-3.5-turbo"],
                "model_ids": {
                    "gpt-4": "gpt-4-0125-preview",
                    "gpt-3.5-turbo": "gpt-3.5-turbo-0125"
                }
            },
            "Named Entity Recognition": {
                "models": ["gpt-4", "gpt-3.5-turbo"],
                "model_ids": {
                    "gpt-4": "gpt-4-0125-preview",
                    "gpt-3.5-turbo": "gpt-3.5-turbo-0125"
                }
            },
            "Summarization": {
                "models": ["gpt-4", "gpt-3.5-turbo"],
                "model_ids": {
                    "gpt-4": "gpt-4-0125-preview",
                    "gpt-3.5-turbo": "gpt-3.5-turbo-0125"
                }
            },
            "Translation": {
                "models": ["gpt-4", "gpt-3.5-turbo"],
                "model_ids": {
                    "gpt-4": "gpt-4-0125-preview",
                    "gpt-3.5-turbo": "gpt-3.5-turbo-0125"
                }
            }
        },
        "Anthropic": {
            "Text Generation": {
                "models": ["claude-3-opus", "claude-3-sonnet", "claude-3-haiku"],
                "model_ids": {
                    "claude-3-opus": "claude-3-opus-20240229",
                    "claude-3-sonnet": "claude-3-sonnet-20240229",
                    "claude-3-haiku": "claude-3-haiku-20240307"
                }
            },
            "Text Classification": {
                "models": ["claude-3-opus", "claude-3-sonnet"],
                "model_ids": {
                    "claude-3-opus": "claude-3-opus-20240229",
                    "claude-3-sonnet": "claude-3-sonnet-20240229"
                }
            },
            "Named Entity Recognition": {
                "models": ["claude-3-opus", "claude-3-sonnet"],
                "model_ids": {
                    "claude-3-opus": "claude-3-opus-20240229",
                    "claude-3-sonnet": "claude-3-sonnet-20240229"
                }
            },
            "Summarization": {
                "models": ["claude-3-opus", "claude-3-sonnet"],
                "model_ids": {
                    "claude-3-opus": "claude-3-opus-20240229",
                    "claude-3-sonnet": "claude-3-sonnet-20240229"
                }
            },
            "Translation": {
                "models": ["claude-3-opus", "claude-3-sonnet"],
                "model_ids": {
                    "claude-3-opus": "claude-3-opus-20240229",
                    "claude-3-sonnet": "claude-3-sonnet-20240229"
                }
            }
        },
        #"HuggingFace": {
        #    "Text Generation": {
        #        "models": ["meta-llama/Llama-2-70b", "mistralai/Mixtral-8x7B", "tiiuae/falcon-40b"],
        #        "model_ids": {
        #            "meta-llama/Llama-2-70b": "meta-llama/Llama-2-70b-chat-hf",
        #            "mistralai/Mixtral-8x7B": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        #            "tiiuae/falcon-40b": "tiiuae/falcon-40b-instruct"
        #        }
        #    },
        #    "Text Classification": {
        #        "models": ["roberta-large", "bert-large-uncased", "xlnet-large-cased"],
        #        "model_ids": {
        #            "roberta-large": "roberta-large",
        #            "bert-large-uncased": "bert-large-uncased",
        #            "xlnet-large-cased": "xlnet-large-cased"
        #        }
        #    },
        #    "Named Entity Recognition": {
        #        "models": ["jean-baptiste/camembert-ner", "dslim/bert-base-NER"],
        #        "model_ids": {
        #            "jean-baptiste/camembert-ner": "jean-baptiste/camembert-ner",
        #            "dslim/bert-base-NER": "dslim/bert-base-NER"
        #        }
        #    },
        #    "Summarization": {
        #        "models": ["facebook/bart-large-cnn", "google/pegasus-large"],
        #        "model_ids": {
        #            "facebook/bart-large-cnn": "facebook/bart-large-cnn",
        #            "google/pegasus-large": "google/pegasus-large"
        #        }
        #    },
        #    "Translation": {
        #        "models": ["Helsinki-NLP/opus-mt-en-fr", "facebook/mbart-large-50"],
        #        
    }

    TASK_DESCRIPTIONS = {
        "Text Generation": "Generate human-like text based on a prompt",
        "Text Classification": "Classify text into predefined categories",
        "Named Entity Recognition": "Identify and classify named entities in text",
        "Summarization": "Create concise summaries of longer texts",
        "Translation": "Translate text between different languages"
    }

    TASK_SETTINGS = {
        "default": {
            "temperature": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 0.7,
                "description": "Controls randomness in the output"
            },
            "max_tokens": {
                "type": "int",
                "min": 50,
                "max": 2000,
                "default": 500,
                "description": "Maximum number of tokens to generate"
            },
            "top_p": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "default": 1.0,
                "description": "Controls diversity via nucleus sampling"
            }
        },
        "Text Generation": {
            "frequency_penalty": {
                "type": "float",
                "min": 0.0,
                "max": 2.0,
                "default": 0.0,
                "description": "Reduces repetition of token sequences"
            },
            "presence_penalty": {
                "type": "float",
                "min": 0.0,
                "max": 2.0,
                "default": 0.0,
                "description": "Reduces repetition of topics"
            }
        },
        "Summarization": {
            "min_length": {
                "type": "int",
                "min": 10,
                "max": 100,
                "default": 30,
                "description": "Minimum length of the summary"
            },
            "max_length": {
                "type": "int",
                "min": 100,
                "max": 500,
                "default": 130,
                "description": "Maximum length of the summary"
            }
        },
        "Translation": {
            "source_lang": {
                "type": "select",
                "options": ["English", "French", "Spanish", "German"],
                "default": "English",
                "description": "Source language"
            },
            "target_lang": {
                "type": "select",
                "options": ["French", "English", "Spanish", "German"],
                "default": "French",
                "description": "Target language"
            }
        }
    }

    @staticmethod
    def process_task(task_type: str, model_provider: str, model: str, 
                        input_text: str, settings: Dict[str, Any], api_key: str = None):
        if not input_text:
            raise ValueError("Please provide input text")

        if model_provider == "OpenAI":
            return ModelNLP._process_openai(task_type, model, input_text, settings, api_key)
        elif model_provider == "Anthropic":
            return ModelNLP._process_anthropic(task_type, model, input_text, settings, api_key)
        else:
            raise NotImplementedError(f"Provider {model_provider} not implemented yet")

    @staticmethod
    def _process_openai(task_type: str, model: str, input_text: str, 
                        settings: Dict[str, Any], api_key: str) -> Any:
        client = OpenAI(api_key=api_key)
        
        # Prepare system messages based on task
        system_messages = {
            "Text Generation": "You are a helpful assistant.",
            "Text Classification": "Classify the following text and return a JSON with 'label' and 'confidence' fields.",
            "Named Entity Recognition": "Identify named entities in the text and return them as a list of JSON objects with 'text' and 'label' fields.",
            "Summarization": "Summarize the following text concisely.",
            "Translation": f"Translate the following text from {settings.get('source_lang', 'English')} to {settings.get('target_lang', 'French')}."
        }

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_messages[task_type]},
                    {"role": "user", "content": input_text}
                ],
                temperature=settings.get("temperature", 0.7),
                max_tokens=settings.get("max_tokens", 500),
                top_p=settings.get("top_p", 1.0),
                frequency_penalty=settings.get("frequency_penalty", 0.0),
                presence_penalty=settings.get("presence_penalty", 0.0)
            )
            
            result = response.choices[0].message.content
            
            # Parse results based on task type
            if task_type == "Text Classification":
                return eval(result)  # Convert string JSON to dict
            elif task_type == "Named Entity Recognition":
                return eval(result)  # Convert string JSON to list
            else:
                return result

        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

    @staticmethod
    def _process_anthropic(task_type: str, model: str, input_text: str, 
                            settings: Dict[str, Any], api_key: str) -> Any:
        client = Anthropic(api_key=api_key)
        
        # Prepare system prompts based on task
        system_prompts = {
            "Text Generation": "Please respond to the following:",
            "Text Classification": "Classify the following text. Return your response as a JSON object with 'label' and 'confidence' fields.",
            "Named Entity Recognition": "Identify named entities in the following text. Return your response as a list of JSON objects, each with 'text' and 'label' fields.",
            "Summarization": "Please provide a concise summary of the following text:",
            "Translation": f"Translate the following text from {settings.get('source_lang', 'English')} to {settings.get('target_lang', 'French')}:"
        }

        try:
            response = client.messages.create(
                model=model,
                max_tokens=settings.get("max_tokens", 500),
                temperature=settings.get("temperature", 0.7),
                top_p=settings.get("top_p", 1.0),
                messages=[{
                    "role": "user",
                    "content": f"{system_prompts[task_type]}\n\n{input_text}"
                }]
            )
            
            result = response.content[0].text
            
            # Parse results based on task type
            if task_type == "Text Classification":
                return eval(result)  # Convert string JSON to dict
            elif task_type == "Named Entity Recognition":
                return eval(result)  # Convert string JSON to list
            else:
                return result

        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")

