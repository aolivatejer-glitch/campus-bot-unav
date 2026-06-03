from typing import Protocol


class LLMProvider(Protocol):
    @property
    def provider_name(self) -> str:
        ...

    @property
    def model_name(self) -> str:
        ...

    def generate(self, prompt: str) -> str:
        ...
