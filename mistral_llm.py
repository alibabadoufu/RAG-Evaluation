"""
Implementation of Mistral LLM inference strategy.
"""
from typing import List, Dict, Any, Optional
import os
import requests

from src.components.base.base_llm import BaseLLM


class MistralLLM(BaseLLM):
    """
    LLM inference strategy using Mistral models.
    
    This LLM uses the Mistral API to generate text responses.
    """
    
    def __init__(self, 
                 model_name: str = "mistral-large-latest",
                 api_key: Optional[str] = None,
                 temperature: float = 0.7, 
                 max_tokens: int = 1024,
                 **kwargs):
        """
        Initialize the Mistral LLM.
        
        Args:
            model_name: The name of the Mistral model to use
            api_key: Mistral API key (if None, will look for MISTRAL_API_KEY env var)
            temperature: Controls randomness in generation (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            **kwargs: Additional parameters for the Mistral API
        """
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens, **kwargs)
        
        # Set up API key
        if api_key is not None:
            self.api_key = api_key
        elif os.environ.get("MISTRAL_API_KEY") is not None:
            self.api_key = os.environ.get("MISTRAL_API_KEY")
        else:
            raise ValueError("Mistral API key must be provided either as an argument or as an environment variable")
        
    def generate(self, prompt: str, context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate text based on a prompt and optional context.
        
        Args:
            prompt: The input prompt for the LLM
            context: Optional list of context documents, each a dictionary with 'text' and 'metadata'
            
        Returns:
            The generated text response
        """
        # Prepare the full prompt with context if provided
        full_prompt = prompt
        if context:
            context_text = "\n\n".join([doc["text"] for doc in context])
            full_prompt = f"Context information:\n{context_text}\n\n{prompt}"
            
        # Call Mistral API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": full_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Mistral API error: {response.status_code} - {response.text}")
    
    def generate_with_chat_history(self, 
                                  messages: List[Dict[str, str]], 
                                  context: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Generate text based on a chat history and optional context.
        
        Args:
            messages: List of message dictionaries, each with 'role' (user/assistant/system) and 'content'
            context: Optional list of context documents, each a dictionary with 'text' and 'metadata'
            
        Returns:
            The generated text response
        """
        # Format the chat history for Mistral API
        formatted_messages = []
        
        # Add context as system message if provided
        if context:
            context_text = "\n\n".join([doc["text"] for doc in context])
            formatted_messages.append({
                "role": "system",
                "content": f"You have access to the following context information:\n{context_text}"
            })
        
        # Add the rest of the messages
        formatted_messages.extend(messages)
        
        # Call Mistral API
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model_name,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        response = requests.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code == 200:
            result = response.json()
            return result["choices"][0]["message"]["content"]
        else:
            raise Exception(f"Mistral API error: {response.status_code} - {response.text}")
