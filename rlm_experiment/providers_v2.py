"""
LLM Providers v2 - Using Native Ollama Client

Prefers ollama.Client over OpenAI-compat for stability.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

try:
    import ollama
    OLLAMA_NATIVE_AVAILABLE = True
except ImportError:
    OLLAMA_NATIVE_AVAILABLE = False


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_s: float
    model: str


class OllamaNativeClient:
    """
    Native Ollama client using ollama.Client.
    More stable than OpenAI-compat endpoint.
    """
    
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "qwen3-next:latest",
        temperature: float = 0.0,
        max_tokens: int = 4096,
        timeout: float = 600.0,
        seed: Optional[int] = None,
    ):
        if not OLLAMA_NATIVE_AVAILABLE:
            raise ImportError("ollama library not installed. Run: pip install ollama")
        
        self.host = host
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.seed = seed
        
        self.client = ollama.Client(host=host, timeout=timeout)
    
    def completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> LLMResponse:
        """
        Send chat completion request.
        
        Args:
            messages: List of {"role": "system"|"user"|"assistant", "content": str}
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            seed: Override default seed for reproducibility
        """
        start_time = time.time()
        
        options = {
            "temperature": temperature if temperature is not None else self.temperature,
            "num_predict": max_tokens if max_tokens is not None else self.max_tokens,
        }
        
        # Add seed if specified
        effective_seed = seed if seed is not None else self.seed
        if effective_seed is not None:
            options["seed"] = effective_seed
        
        response = self.client.chat(
            model=self.model,
            messages=messages,
            options=options,
        )
        
        latency = time.time() - start_time
        
        return LLMResponse(
            content=response.get("message", {}).get("content", ""),
            prompt_tokens=response.get("prompt_eval_count", 0),
            completion_tokens=response.get("eval_count", 0),
            total_tokens=response.get("prompt_eval_count", 0) + response.get("eval_count", 0),
            latency_s=latency,
            model=self.model,
        )
    
    def simple_completion(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> str:
        """Simple single-turn completion."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.completion(messages, temperature=temperature, seed=seed)
        return response.content


def get_client(
    host: str = None,
    model: str = None,
    temperature: float = 0.0,
    seed: int = None,
) -> OllamaNativeClient:
    """Create client from environment or arguments."""
    return OllamaNativeClient(
        host=host or os.environ.get("OLLAMA_HOST", "http://ece-nebula04.eng.uwaterloo.ca:11434"),
        model=model or os.environ.get("OLLAMA_MODEL", "qwen3-next:latest"),
        temperature=temperature,
        seed=seed,
    )


# Test
if __name__ == "__main__":
    client = get_client()
    response = client.simple_completion("Say 'hello' and nothing else.")
    print(f"Response: {response}")
