from typing import List, Dict, Optional, Any
from openai import OpenAI
import logging
import time

logger = logging.getLogger(__name__)

class OpenAILLM:
    """OpenAI API implementation"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-nano", # need you to change this to the model you want to use
        system_instruction: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ):
        """
        Initialize OpenAI interface
        
        Args:
            api_key: API key
            model: Model name, default is gpt-4.1-nano, need you to change this to the model you want to use
            system_instruction: System instruction
            generation_config: Generation configuration dictionary, contains the following optional parameters:
                - temperature: float = 0.7
                - max_tokens: int = 1024
                - top_p: float = 1.0
                - frequency_penalty: float = 0.0
                - presence_penalty: float = 0.0
            max_retries: Maximum number of retries
            retry_delay: Retry interval time (seconds)
        """
        self.api_key = api_key
        self.model = model
        self.system_instruction = system_instruction
        self.generation_config = generation_config or {}
        self.client = OpenAI(api_key=api_key)
        # Retry strategy configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
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
            # Use the incoming system_prompt first, then use the system_instruction set during initialization
            system_content = system_prompt or self.system_instruction
            if system_content:
                messages.append({"role": "system", "content": system_content})
            messages.append({"role": "user", "content": prompt})
            
            # Use instance configuration to override default values
            config_params = {
                'temperature': temperature or self.generation_config.get('temperature', 0.7),
                'max_tokens': max_tokens or self.generation_config.get('max_tokens', 150),
                **kwargs
            }
            
            # If there are other parameters in generation_config, also add them in
            for key, value in self.generation_config.items():
                if key not in ['temperature', 'max_tokens'] and key not in config_params:
                    config_params[key] = value
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **config_params
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI error: {str(e)}")
            return ""
            
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        Generate reply based on multi-turn conversation
        
        Args:
            messages: Message list
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens to generate
            **kwargs: Other parameters
            
        Returns:
            str: Generated reply
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                # Use instance configuration to override default values
                config_params = {
                    'temperature': temperature or self.generation_config.get('temperature', 0.7),
                    'max_tokens': max_tokens or self.generation_config.get('max_tokens', 150),
                    **kwargs
                }
                
                # If there are other parameters in generation_config, also add them in
                for key, value in self.generation_config.items():
                    if key not in ['temperature', 'max_tokens'] and key not in config_params:
                        config_params[key] = value
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    **config_params
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                last_exception = e
                error_msg = str(e)
                
                # Check if it is a retryable error
                if any(code in error_msg for code in ["503", "502", "504", "500"]):  # Server error
                    if attempt < self.max_retries:
                        logger.warning(f"OpenAI API server error (attempt {attempt + 1}/{self.max_retries + 1}): {error_msg}")
                        logger.info(f"Waiting {self.retry_delay} seconds before retrying...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        logger.error(f"OpenAI API server error, maximum retries reached {self.max_retries}: {error_msg}")
                elif "429" in error_msg:  # Rate limit error
                    if attempt < self.max_retries:
                        logger.warning(f"OpenAI API rate limit (attempt {attempt + 1}/{self.max_retries + 1}): {error_msg}")
                        logger.info(f"Waiting {self.retry_delay * 2} seconds before retrying...")
                        time.sleep(self.retry_delay * 2)  # Wait longer when rate limited
                        continue
                    else:
                        logger.error(f"OpenAI API rate limit, maximum retries reached {self.max_retries}: {error_msg}")
                else:
                    # Other types of errors, no retries
                    logger.error(f"OpenAI conversation error: {error_msg}")
                    break
        
        # If all retries fail, record the last exception and return an empty string
        logger.error(f"OpenAI conversation failed, all retries exhausted: {str(last_exception)}")
        return "" 