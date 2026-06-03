from rag_chatbot.llm.base import LLMProvider


class GeminiProvider(LLMProvider):
    def __init__(
        self,
        *,
        api_key: str,
        model_name: str,
        temperature: float = 0.2,
    ) -> None:
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required when Gemini mode is enabled.")

        try:
            from google import genai
        except ImportError as exc:
            raise RuntimeError(
                "google-genai is not installed. Install/update dependencies with: "
                "python -m pip install -e ."
            ) from exc

        self._client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._temperature = temperature

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(self, prompt: str) -> str:
        response = self._client.models.generate_content(
            model=self._model_name,
            contents=prompt,
            config={"temperature": self._temperature},
        )
        text = getattr(response, "text", None)
        return "" if text is None else str(text).strip()
