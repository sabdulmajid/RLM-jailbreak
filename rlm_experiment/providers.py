"""
LLM Provider Adapters - OpenAI-compatible clients for various backends.
Supports: OpenAI, Ollama (via OpenAI-compatible endpoint), vLLM, LiteLLM, etc.
"""

import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from openai import OpenAI
import time

@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    model: str

class BaseLLMClient:
    """Base class for LLM clients."""
    
    def completion(self, messages: List[Dict[str, str]], **kwargs) -> LLMResponse:
        raise NotImplementedError
    
    def simple_completion(self, prompt: str, **kwargs) -> str:
        """Convenience method for simple string prompts."""
        messages = [{"role": "user", "content": prompt}]
        response = self.completion(messages, **kwargs)
        return response.content

class OpenAICompatibleClient(BaseLLMClient):
    """
    OpenAI-compatible client that works with:
    - OpenAI API
    - Ollama (http://host:11434/v1)
    - vLLM (http://host:8000/v1)
    - LiteLLM proxy
    - Any OpenAI-compatible endpoint
    """
    
    def __init__(
        self,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 600.0,
        seed: Optional[int] = None,
    ):
        self.api_base = api_base or os.getenv("API_BASE", "https://api.openai.com/v1")
        self.api_key = api_key or os.getenv("API_KEY", os.getenv("OPENAI_API_KEY", "ollama"))
        self.model = model or os.getenv("MODEL", "gpt-4")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.seed = seed
        
        # Create OpenAI client with custom base URL
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
            timeout=self.timeout,
        )
    
    def completion(
        self,
        messages: List[Dict[str, str]] | str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Make a completion request."""
        
        # Handle string input
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        
        start_time = time.time()
        
        try:
            # Build request params
            request_params = {
                "model": self.model,
                "messages": messages,
                "temperature": temperature if temperature is not None else self.temperature,
            }
            
            # Add max_tokens if specified
            if max_tokens or self.max_tokens:
                request_params["max_tokens"] = max_tokens or self.max_tokens
            
            # Add seed if supported and specified
            if self.seed is not None:
                request_params["seed"] = self.seed
            
            response = self.client.chat.completions.create(**request_params)
            
            latency_ms = (time.time() - start_time) * 1000
            
            # Extract usage info (may not be available for all providers)
            usage = response.usage
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            
            return LLMResponse(
                content=response.choices[0].message.content or "",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=latency_ms,
                model=self.model,
            )
            
        except Exception as e:
            latency_ms = (time.time() - start_time) * 1000
            return LLMResponse(
                content=f"[ERROR] {str(e)}",
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                latency_ms=latency_ms,
                model=self.model,
            )

class OllamaClient(OpenAICompatibleClient):
    """
    Convenience class for Ollama endpoints.
    Ollama exposes OpenAI-compatible API at /v1/chat/completions
    """
    
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen3-next:latest",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        num_ctx: int = 125000,
        timeout: float = 600.0,
        seed: Optional[int] = None,
    ):
        # Ollama's OpenAI-compatible endpoint is at /v1
        api_base = f"{host.rstrip('/')}/v1"
        
        super().__init__(
            api_base=api_base,
            api_key="ollama",  # Ollama doesn't require a real key
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            seed=seed,
        )
        
        self.num_ctx = num_ctx
        self.host = host

def get_client_from_env() -> OpenAICompatibleClient:
    """
    Create a client from environment variables.
    
    Environment variables:
        API_BASE: Base URL for the API (default: https://api.openai.com/v1)
        API_KEY: API key (default: OPENAI_API_KEY or "ollama")
        MODEL: Model name (default: gpt-4)
        SUB_MODEL: Model for sub-calls (default: same as MODEL)
        TEMPERATURE: Temperature (default: 0.0)
        MAX_TOKENS: Max tokens (default: 4096)
        SEED: Random seed for reproducibility (optional)
    """
    return OpenAICompatibleClient(
        api_base=os.getenv("API_BASE"),
        api_key=os.getenv("API_KEY"),
        model=os.getenv("MODEL", "gpt-4"),
        temperature=float(os.getenv("TEMPERATURE", "0.0")),
        max_tokens=int(os.getenv("MAX_TOKENS", "4096")),
        seed=int(os.getenv("SEED")) if os.getenv("SEED") else None,
    )

# Test
if __name__ == "__main__":
    # Test Ollama connection
    client = OllamaClient(
        host="http://ece-nebula04.eng.uwaterloo.ca:11434",
        model="qwen3-next:latest",
    )
    
    response = client.simple_completion("Say 'Hello World' and nothing else.")
    print(f"Response: {response}")
