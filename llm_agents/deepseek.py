from typing import List, Dict, Optional, Any
from openai import OpenAI
import logging

logger = logging.getLogger(__name__)

class DeepseekLLM:
    """Deepseek API implementation"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        system_instruction: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Deepseek interface
        
        Args:
            api_key: API key
            model: Model name
            system_instruction: System instruction
            generation_config: Generation configuration dictionary, contains the following optional parameters:
                - temperature: float = 0.7
                - max_tokens: int = 1024
                - top_p: float = 1.0
                - stream: bool = False
        """
        self.api_key = api_key
        self.model = model
        self.system_instruction = system_instruction
        self.generation_config = generation_config or {}
        
        # Use OpenAI client, but set Deepseek's base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate text response
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (if provided, it will override the system_instruction set during initialization)
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens to generate
            **kwargs: Other parameters
            
        Returns:
            str: Generated text response
        """
        try:
            messages = []
            # Use the system_prompt passed in first, then use the system_instruction set during initialization
            system_content = system_prompt or self.system_instruction
            if system_content:
                messages.append({"role": "system", "content": system_content})
            messages.append({"role": "user", "content": prompt})
            
            # Use instance configuration to override default values
            config_params = {
                'temperature': temperature or self.generation_config.get('temperature', 0.7),
                'max_tokens': max_tokens or self.generation_config.get('max_tokens', 1024),
                'stream': self.generation_config.get('stream', False),
                **kwargs
            }
            
            # If there are other parameters in generation_config, also add them in
            for key, value in self.generation_config.items():
                if key not in ['temperature', 'max_tokens', 'stream'] and key not in config_params:
                    config_params[key] = value
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **config_params
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Deepseek generation error: {str(e)}")
            return ""
            
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Perform conversation
        
        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens to generate
            **kwargs: Other parameters
            
        Returns:
            str: Generated reply
        """
        try:
            # Use instance configuration to override default values
            config_params = {
                'temperature': temperature or self.generation_config.get('temperature', 0.7),
                'max_tokens': max_tokens or self.generation_config.get('max_tokens', 1024),
                'stream': self.generation_config.get('stream', False),
                **kwargs
            }
            
            # If there are other parameters in generation_config, also add them in
            for key, value in self.generation_config.items():
                if key not in ['temperature', 'max_tokens', 'stream'] and key not in config_params:
                    config_params[key] = value
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **config_params
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Deepseek conversation error: {str(e)}")
            return "" 