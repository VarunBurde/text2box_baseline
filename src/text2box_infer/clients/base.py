from __future__ import annotations

from abc import ABC, abstractmethod

from ..config import Settings
from ..types import ModelRequest


class VisionProvider(ABC):
    name: str

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abstractmethod
    def predict(self, image_bytes: bytes, request: ModelRequest) -> str:
        """Run model inference and return text output."""

def create_provider(provider: str, settings: Settings) -> VisionProvider:
    provider_norm = provider.lower()

    if provider_norm == "openai":
        from .openai_client import OpenAIProvider
        return OpenAIProvider(settings)

    if provider_norm == "ollama":
        from .openai_client import OllamaProvider
        return OllamaProvider(settings)

    raise ValueError(f"Unsupported provider: {provider!r}. Choose 'openai' or 'ollama'.")
