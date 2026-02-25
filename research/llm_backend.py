#!/usr/bin/env python3
"""
llm_backend.py — Unified LLM interface for research experiments

Supports:
- Ollama (local models)
- Groq (cloud API)
- Mock (testing)
"""

import os
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class LLMConfig:
    """LLM configuration."""
    backend: str  # "ollama", "groq", "mock"
    model: str
    temperature: float = 0.7
    max_tokens: int = 1000
    base_url: Optional[str] = None  # For Ollama: http://localhost:11434


class LLMBackend:
    """Unified LLM interface."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.backend = self._init_backend()
    
    def _init_backend(self):
        """Initialize backend based on config."""
        if self.config.backend == "ollama":
            return OllamaBackend(self.config)
        elif self.config.backend == "groq":
            return GroqBackend(self.config)
        elif self.config.backend == "mock":
            return MockBackend(self.config)
        else:
            raise ValueError(f"Unknown backend: {self.config.backend}")
    
    def ask(self, system: str, prompt: str, temperature: Optional[float] = None) -> str:
        """Ask LLM a question."""
        temp = temperature if temperature is not None else self.config.temperature
        return self.backend.ask(system, prompt, temp)
    
    def __call__(self, system: str, prompt: str, temperature: float = 0.7) -> str:
        """Callable interface for compatibility."""
        return self.ask(system, prompt, temperature)


class OllamaBackend:
    """Ollama local backend."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.base_url = config.base_url or "http://localhost:11434"
        
        # Try to import requests
        try:
            import requests
            self.requests = requests
        except ImportError:
            raise ImportError("requests required for Ollama. Install: pip install requests")
    
    def ask(self, system: str, prompt: str, temperature: float) -> str:
        """Ask Ollama model."""
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt}
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": self.config.max_tokens,
            }
        }
        
        try:
            response = self.requests.post(url, json=payload, timeout=120)  # Increased timeout
            
            if response.status_code == 500:
                print(f"⚠️  Ollama 500 error - may need to restart ollama service")
                print(f"   Try: pkill ollama && ollama serve")
                return None
            
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"].strip()
        except self.requests.exceptions.Timeout:
            print(f"⚠️  Ollama timeout - model may be too large or slow")
            return None
        except Exception as e:
            print(f"⚠️  Ollama error: {e}")
            return None


class GroqBackend:
    """Groq cloud backend."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        
        try:
            from groq import Groq
            self.client = Groq()
        except ImportError:
            raise ImportError("groq package required. Install: pip install groq")
    
    def ask(self, system: str, prompt: str, temperature: float) -> str:
        """Ask Groq model."""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=self.config.max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️  Groq error: {e}")
            return None


class MockBackend:
    """Mock backend for testing."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def ask(self, system: str, prompt: str, temperature: float) -> str:
        """Return mock response."""
        # Parse question type from prompt
        if "capital of France" in prompt.lower():
            return "CHOICE: Paris\nCONFIDENCE: 1.00\nRATIONALE: Paris is the capital of France"
        elif "15% of 200" in prompt:
            return "CHOICE: 30\nCONFIDENCE: 0.95\nRATIONALE: 15% of 200 = 30"
        elif "dangerous" in prompt.lower() or "regulate" in prompt.lower():
            return "CHOICE: It depends\nCONFIDENCE: 0.60\nRATIONALE: Context-dependent question"
        else:
            return "CHOICE: A\nCONFIDENCE: 0.70\nRATIONALE: Mock response for testing"


# Convenience factory functions

def create_ollama_backend(model: str = "mistral-large-3:675b-cloud", temperature: float = 0.7) -> LLMBackend:
    """Create Ollama backend (recommended for research).
    
    Default: mistral-large-3:675b-cloud — high-quality reasoning via cloud routing
    Alternative: tinyllama (637 MB) — fast, works on low-RAM systems
    Alternative: phi3:mini (2.2 GB) — needs ~3.5 GB RAM
    """
    config = LLMConfig(
        backend="ollama",
        model=model,
        temperature=temperature,
        max_tokens=1000,
    )
    return LLMBackend(config)


def create_groq_backend(model: str = "mixtral-8x7b-32768", temperature: float = 0.7) -> LLMBackend:
    """Create Groq backend."""
    config = LLMConfig(
        backend="groq",
        model=model,
        temperature=temperature,
        max_tokens=1000,
    )
    return LLMBackend(config)


def create_mock_backend() -> LLMBackend:
    """Create mock backend for testing."""
    config = LLMConfig(
        backend="mock",
        model="mock",
    )
    return LLMBackend(config)


# Auto-detect best available backend

def auto_backend(prefer_local: bool = True) -> LLMBackend:
    """Auto-detect and create best available backend.

    Env vars:
      COGNOS_BACKEND   — "ollama" | "groq" (overrides auto-detect)
      COGNOS_MODEL     — model name for the selected backend
      COGNOS_TEMPERATURE — sampling temperature (default 0.7)

    Priority (if COGNOS_BACKEND not set and prefer_local=True):
    1. Ollama (if available)
    2. Groq (if GROQ_API_KEY set)
    3. Mock (fallback)
    """
    model_override = os.getenv("COGNOS_MODEL")
    temp_override = os.getenv("COGNOS_TEMPERATURE")
    temperature = float(temp_override) if temp_override else 0.7
    backend_override = os.getenv("COGNOS_BACKEND", "").lower()

    # Explicit backend selection
    if backend_override == "groq":
        model = model_override or "llama-3.3-70b-versatile"
        backend = create_groq_backend(model=model, temperature=temperature)
        print(f"✓ Using Groq ({model}, temperature={temperature})")
        return backend

    if backend_override == "ollama" or prefer_local:
        try:
            model = model_override or "mistral-large-3:675b-cloud"
            backend = create_ollama_backend(model=model, temperature=temperature)
            test = backend.ask("You are helpful.", "Say 'OK'", 0.0)
            if test:
                print(f"✓ Using Ollama ({model}, temperature={temperature})")
                return backend
        except Exception:
            pass

    # Try Groq as fallback
    if os.getenv("GROQ_API_KEY"):
        try:
            model = model_override or "llama-3.3-70b-versatile"
            backend = create_groq_backend(model=model, temperature=temperature)
            print(f"✓ Using Groq ({model}, temperature={temperature})")
            return backend
        except Exception:
            pass

    print("⚠️  Using mock backend (Ollama/Groq not available)")
    return create_mock_backend()


if __name__ == '__main__':
    # Test backends
    print("Testing LLM backends...\n")
    
    # Test Ollama
    print("1. Testing Ollama (qwen2.5:7b):")
    try:
        llm = create_ollama_backend("qwen2.5:7b")
        response = llm.ask("You are helpful.", "What is 2+2?", 0.0)
        print(f"   Response: {response[:100] if response else 'FAILED'}\n")
    except Exception as e:
        print(f"   Error: {e}\n")
    
    # Test auto-detect
    print("2. Testing auto-detect:")
    llm = auto_backend(prefer_local=True)
    response = llm.ask("You are helpful.", "Say hello", 0.0)
    print(f"   Response: {response[:100] if response else 'FAILED'}")
