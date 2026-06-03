import os

from rag_chatbot.embeddings.base import EmbeddingProvider


class SentenceTransformersEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        *,
        model_name: str,
        device: str = "cpu",
        batch_size: int = 16,
        normalize_embeddings: bool = True,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._batch_size = batch_size
        self._normalize_embeddings = normalize_embeddings

        _configure_local_embedding_runtime()

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. "
                "Install project dependencies with: python -m pip install -e ."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                "sentence-transformers could not be imported. This project uses "
                "local embeddings with PyTorch; if Transformers tries to load "
                "TensorFlow/Keras, set these variables before running the command: "
                "set USE_TF=0; set TF_CPP_MIN_LOG_LEVEL=3; "
                "set TF_ENABLE_ONEDNN_OPTS=0"
            ) from exc

        try:
            self._model = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            raise RuntimeError(
                "Could not load the local embedding model. "
                "The first run may need internet access to download model weights "
                "from Hugging Face; document contents are not sent by this step. "
                "If the error mentions TensorFlow or Keras, set: "
                "set USE_TF=0; set TF_CPP_MIN_LOG_LEVEL=3; "
                "set TF_ENABLE_ONEDNN_OPTS=0"
            ) from exc

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def device(self) -> str:
        return self._device

    @property
    def dimension(self) -> int | None:
        return self._model.get_sentence_embedding_dimension()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        return self._encode([self._passage_text(text) for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self._query_text(text)])[0]

    @staticmethod
    def _passage_text(text: str) -> str:
        return f"passage: {text}"

    @staticmethod
    def _query_text(text: str) -> str:
        return f"query: {text}"

    def _encode(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        if hasattr(embeddings, "tolist"):
            return embeddings.tolist()

        return [list(embedding) for embedding in embeddings]


def _configure_local_embedding_runtime() -> None:
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
