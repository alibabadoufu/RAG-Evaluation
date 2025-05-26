"""
Implementation of Llama LLM inference strategy.
"""
from typing import List, Dict, Any, Optional
import os
from llama_cpp import Llama

from src.components.base.base_llm import BaseLLM


class LlamaLLM(BaseLLM):
    """
    LLM inference strategy using Llama models.
    
    This LLM uses the llama-cpp-python library to run Llama models locally.
    """
    
    def __init__(self, 
                 model_path: str,
                 model_name: str = "llama-3.1",
                 temperature: float = 0.7, 
                 max_tokens: int = 1024,
                 context_window: int = 4096,
                 **kwargs):
        """
        Initialize the Llama LLM.
        
        Args:
            model_path: Path to the Llama model file
            model_name: Name identifier for the model
            temperature: Controls randomness in generation (0.0 to 1.0)
            max_tokens: Maximum number of tokens to generate
            context_window: Size of the context window
            **kwargs: Additional parameters for the Llama model
        """
        super().__init__(model_name=model_name, temperature=temperature, max_tokens=max_tokens, **kwargs)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        self.model_path = model_path
        self.context_window = context_window
        
        # Initialize the Llama model
        self.model = Llama(
            model_path=model_path,
            n_ctx=context_window,
            **kwargs
        )
        
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
            
        # Generate response
        response = self.model(
            full_prompt,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["</s>", "<|im_end|>"]  # Common stop tokens for Llama models
        )
        
        return response["choices"][0]["text"]
    
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
        # Format the chat history in Llama chat format
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
        
        # Convert to Llama chat format
        chat_text = ""
        for msg in formatted_messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                chat_text += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                chat_text += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                chat_text += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        # Add the final assistant prompt
        chat_text += "<|im_start|>assistant\n"
        
        # Generate response
        response = self.model(
            chat_text,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=["</s>", "<|im_end|>"]
        )
        
        return response["choices"][0]["text"]
