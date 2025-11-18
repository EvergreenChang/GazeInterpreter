from typing import Optional, Dict, Any
from google import genai
from google.genai import types
import logging
import time

logger = logging.getLogger(__name__)

class GeminiLLM:
    """Google Gemini API implementation"""
    
    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash",
        reasoning_llm: bool = False,
        system_instruction: Optional[str] = None,
        thinking_budget: Optional[int] = None,
        temperature: Optional[float] = None,
        generation_config: Optional[Dict[str, Any]] = None,
        max_retries: int = 5,
        retry_delay: float = 5.0
    ):
        """
        Initialize Gemini interface
        
        Args:
            api_key: API key
            model: Model name
            system_instruction: System instruction
            generation_config: Generation configuration dictionary, contains the following optional parameters:
                - temperature: float = 0.7
                - max_tokens: int = 250
                - top_p: float = 0.8
                - top_k: int = 40
                - candidate_count: int = 1
                - stop_sequences: Optional[list[str]] = None
            max_retries: Maximum number of retries
            retry_delay: Retry interval time (seconds)
        """
        self.api_key = api_key
        self.model = model
        self.reasoning_llm = reasoning_llm
        self.thinking_budget = thinking_budget
        self.temperature = temperature
        self.client = genai.Client(api_key=api_key)
        # Save system instruction
        self.system_instruction = system_instruction
        # Save generation configuration
        self.generation_config = generation_config or {}
        # Retry strategy configuration
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
    def _create_generation_config(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        candidate_count: Optional[int] = None,
        stop_sequences: Optional[list[str]] = None,
        **kwargs
    ) -> types.GenerateContentConfig:
        """
        Create generation configuration
        
        Args:
            temperature: Temperature parameter
            max_tokens: Maximum number of tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            candidate_count: Number of candidates to generate
            stop_sequences: Stop sequences
            **kwargs: Other parameters
            
        Returns:
            GenerateContentConfig: Generation configuration object
        """
        # Use instance configuration to override default values
        config_params = {
            'temperature': temperature or self.temperature,
            'max_output_tokens': max_tokens or self.generation_config.get('max_tokens', 250),
            'top_p': top_p or self.generation_config.get('top_p', 0.8),
            'top_k': top_k or self.generation_config.get('top_k', 40),
            'candidate_count': candidate_count or self.generation_config.get('candidate_count', 1),
            'stop_sequences': stop_sequences or self.generation_config.get('stop_sequences'),
            **kwargs
        }
        
        # Remove None values
        config_params = {k: v for k, v in config_params.items() if v is not None}
        
        config = types.GenerateContentConfig(**config_params)
        
        # If there is a system instruction, add it to the configuration
        if self.system_instruction:
            config.system_instruction = self.system_instruction
            
        return config
        
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
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
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if self.reasoning_llm:
                # If system_prompt is provided, create a new configuration
                    if system_prompt or self.system_instruction:
                        generation_config = types.GenerateContentConfig(
                            system_instruction=system_prompt or self.system_instruction,
                            temperature=temperature or self.temperature,
                            thinking_config=types.ThinkingConfig(thinking_budget=self.thinking_budget) # Disables thinking
                        )
                    else:
                        generation_config = self._create_generation_config(
                            temperature=temperature,
                            max_tokens=max_tokens,
                        )
                else:
                      generation_config = types.GenerateContentConfig(
                            system_instruction=system_prompt or self.system_instruction,
                            temperature=temperature or self.temperature,
                        )
                # Generate response
                response = self.client.models.generate_content(
                    model=self.model,
                    config=generation_config,
                    contents=prompt
                )
                
                # Process response
                if response.candidates:
                    return response.candidates[0].content.parts[0].text.strip()
                return ""
                
            except Exception as e:
                last_exception = e
                error_msg = str(e)
                
                # Check if it is a 503 error (service overload)
                if "503" in error_msg and "overloaded" in error_msg.lower():
                    if attempt < self.max_retries:
                        logger.warning(f"Gemini API service overload (attempt {attempt + 1}/{self.max_retries + 1}): {error_msg}")
                        logger.info(f"Waiting {self.retry_delay} seconds before retrying...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        logger.error(f"Gemini API service overload, maximum retries reached {self.max_retries}: {error_msg}")
                elif "429" in error_msg:  # Rate limit error
                    if attempt < self.max_retries:
                        logger.warning(f"Gemini API rate limit (attempt {attempt + 1}/{self.max_retries + 1}): {error_msg}")
                        logger.info(f"Waiting {self.retry_delay * 2} seconds before retrying...")
                        time.sleep(self.retry_delay * 2)  # Wait longer when rate limited
                        continue
                    else:
                        logger.error(f"Gemini API rate limit, maximum retries reached {self.max_retries}: {error_msg}")
                elif "500" in error_msg or "502" in error_msg or "504" in error_msg:  # Other server errors
                    if attempt < self.max_retries:
                        logger.warning(f"Gemini API server error (attempt {attempt + 1}/{self.max_retries + 1}): {error_msg}")
                        logger.info(f"Waiting {self.retry_delay} seconds before retrying...")
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        logger.error(f"Gemini API server error, maximum retries reached {self.max_retries}: {error_msg}")
                else:
                    # Other types of errors, no retries
                    logger.error(f"Gemini error: {error_msg}")
                    break
        
        # If all retries fail, record the last exception and return an empty string
        logger.error(f"Gemini failed, all retries exhausted: {str(last_exception)}")
        return "" 