"""
LLM Backend Interface

Unified interface for different LLM providers (Aliyun, DeepSeek, etc.)
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

try:
    import openai
except ImportError:
    openai = None

logger = logging.getLogger(__name__)


class LLMBackend(ABC):
    """Abstract base class for LLM backends."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt."""
        pass

    @staticmethod
    def create(backend_type: str, **kwargs) -> "LLMBackend":
        """Factory method to create backend."""
        if backend_type == "aliyun":
            return AliyunBackend(**kwargs)
        elif backend_type == "deepseek":
            return DeepSeekBackend(**kwargs)
        else:
            raise ValueError(f"Unknown backend type: {backend_type}")


class AliyunBackend(LLMBackend):
    """Aliyun Bailian LLM backend."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "kimi-k2.5",
        temperature: float = 0.7,
        max_tokens: int = 2048,
        base_url: str = "https://dashscope.aliyuncs.com/api/v1"
    ):
        self.api_key = api_key or os.environ.get("ALIYUN_API_KEY")
        if not self.api_key:
            raise ValueError("API key required for Aliyun backend")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url

        if openai:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url=base_url
            )
        else:
            self.client = None

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Aliyun API."""
        if not self.client:
            raise RuntimeError("OpenAI client not available")

        max_retries = kwargs.get("max_retries", 3)
        retry_delay = kwargs.get("retry_delay", 1.0)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert neural architecture designer. Generate PyTorch code only."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content

            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    raise

        return ""


class DeepSeekBackend(LLMBackend):
    """DeepSeek API backend."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("API key required for DeepSeek backend")

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        if openai:
            self.client = openai.OpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com"
            )
        else:
            self.client = None

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using DeepSeek API."""
        if not self.client:
            raise RuntimeError("OpenAI client not available")

        max_retries = kwargs.get("max_retries", 3)

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert neural architecture designer."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                return response.choices[0].message.content

            except Exception as e:
                logger.warning(f"DeepSeek API call failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1.0 * (2 ** attempt))
                else:
                    raise

        return ""
