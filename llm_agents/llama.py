from typing import List, Dict, Optional, Any
from openai import OpenAI
import logging
import time
import random
import backoff

logger = logging.getLogger(__name__)

# Define exponential backoff retry decorator to handle 429 errors
@backoff.on_exception(
    backoff.expo,
    Exception,
    max_tries=5,
    max_time=30,
    giveup=lambda e: not (hasattr(e, 'status_code') and e.status_code == 429)
)
def api_call_with_backoff(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if hasattr(e, 'status_code') and e.status_code == 429:
                logger.warning(f"Rate limit (429), performing backoff retry...")
                # Add random delay to avoid multiple requests retrying at the same time
                time.sleep(random.uniform(1.0, 3.0))
                return func(*args, **kwargs)
            raise e
    return wrapper

class LlamaLLM:
    """Bailei platform Llama model API implementation"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "llama-4-maverick-17b-128e-instruct", # need you to change this to the model you want to use
        system_instruction: Optional[str] = None,
        generation_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize Bailei platform Llama interface
        
        Args:
            api_key: Bailei platform API key
            model: Model name, default is llama-4-maverick-17b-128e-instruct, need you to change this to the model you want to use
            system_instruction: System instruction
            generation_config: Generation configuration dictionary, contains the following optional parameters:
                - temperature: float = 0.7
                - max_tokens: int = 1024
                - top_p: float = 1.0
                - frequency_penalty: float = 0.0
                - presence_penalty: float = 0.0
        """
        self.api_key = api_key
        self.model = model
        self.system_instruction = system_instruction
        self.generation_config = generation_config or {}
        
        # Use OpenAI client, but set Bailei platform base_url
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            # Set longer timeout, Bailei platform may need more processing time
            timeout=60.0
        )
        
        # Request interval control
        self.last_request_time = 0
        self.min_request_interval = 7  # Minimum request interval (seconds)
        
    def _wait_for_rate_limit(self):
        """Wait for request interval to avoid triggering rate limit"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        if elapsed < self.min_request_interval:
            # Add some randomness to avoid multiple requests sending at the same time
            sleep_time = self.min_request_interval - elapsed + random.uniform(0.1, 0.5)
            logger.debug(f"Waiting {sleep_time:.2f} seconds to avoid rate limit")
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
        
    @api_call_with_backoff
    def _create_completion(self, model, messages, **config_params):
        """Wrap API call, add retry and rate limit"""
        self._wait_for_rate_limit()
        return self.client.chat.completions.create(
            model=model,
            messages=messages,
            **config_params
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
            # Use the incoming system_prompt first, then use the system_instruction set during initialization
            system_content = system_prompt or self.system_instruction
            if system_content:
                messages.append({"role": "system", "content": system_content})
            messages.append({"role": "user", "content": prompt})
            
            # Use instance configuration to override default values
            config_params = {
                'temperature': temperature or self.generation_config.get('temperature', 0.7),
                'max_tokens': max_tokens or self.generation_config.get('max_tokens', 1024),
                **kwargs
            }
            
            # If there are other parameters in generation_config, also add them in
            for key, value in self.generation_config.items():
                if key not in ['temperature', 'max_tokens'] and key not in config_params:
                    config_params[key] = value
            
            # Use retry mechanism to call API
            max_retries = 3
            for retry in range(max_retries):
                try:
                    response = self._create_completion(
                        model=self.model,
                        messages=messages,
                        **config_params
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    if retry < max_retries - 1:
                        wait_time = (2 ** retry) + random.uniform(0, 1)  # Exponential backoff
                        logger.warning(f"Request failed, waiting {wait_time:.2f} seconds before retrying ({retry+1}/{max_retries}): {str(e)}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Llama error: {str(e)}")
                        return ""
            
        except Exception as e:
            logger.error(f"Llama error: {str(e)}")
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
        try:
            # Use instance configuration to override default values
            config_params = {
                'temperature': temperature or self.generation_config.get('temperature', 0.7),
                'max_tokens': max_tokens or self.generation_config.get('max_tokens', 1024),
                **kwargs
            }
            
            # If there are other parameters in generation_config, also add them in
            for key, value in self.generation_config.items():
                if key not in ['temperature', 'max_tokens'] and key not in config_params:
                    config_params[key] = value
            
            # Use retry mechanism to call API
            max_retries = 3
            for retry in range(max_retries):
                try:
                    response = self._create_completion(
                        model=self.model,
                        messages=messages,
                        **config_params
                    )
                    return response.choices[0].message.content.strip()
                except Exception as e:
                    if retry < max_retries - 1:
                        wait_time = (2 ** retry) + random.uniform(0, 1)  # Exponential backoff
                        logger.warning(f"Request failed, waiting {wait_time:.2f} seconds before retrying ({retry+1}/{max_retries}): {str(e)}")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Llama conversation error: {str(e)}")
                        return ""
            
        except Exception as e:
            logger.error(f"Llama conversation error: {str(e)}")
            return "" 