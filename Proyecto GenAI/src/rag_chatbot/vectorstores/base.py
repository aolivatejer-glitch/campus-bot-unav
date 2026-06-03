from abc import ABC, abstractmethod

from rag_chatbot.schemas import Chunk


class VectorStore(ABC):
    @property
    @abstractmethod
    def collection_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def existing_ids(self, ids: list[str]) -> set[str]:
        raise NotImplementedError

    @abstractmethod
    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def count(self) -> int:
        raise NotImplementedError
