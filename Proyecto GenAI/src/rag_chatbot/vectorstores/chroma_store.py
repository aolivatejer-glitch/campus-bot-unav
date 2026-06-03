from importlib.util import find_spec
from pathlib import Path

from rag_chatbot.schemas import Chunk
from rag_chatbot.vectorstores.base import VectorStore


def is_chromadb_available() -> bool:
    return find_spec("chromadb") is not None


def build_chroma_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    return {
        "document_id": chunk.document_id,
        "file_name": chunk.file_name,
        "file_path": chunk.file_path,
        "file_type": chunk.file_type,
        "page_number": chunk.page_number if chunk.page_number is not None else -1,
        "chunk_index": chunk.chunk_index,
        "char_count": chunk.char_count,
        "source": str(chunk.metadata.get("source", "")),
    }


def build_chroma_where(
    *,
    document_id: str | None = None,
    file_name: str | None = None,
) -> dict[str, object] | None:
    filters: list[dict[str, str]] = []

    if document_id:
        filters.append({"document_id": document_id})
    if file_name:
        filters.append({"file_name": file_name})

    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


class ChromaVectorStore(VectorStore):
    def __init__(self, *, persist_dir: Path, collection_name: str) -> None:
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection_name

        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb is not installed. Install/update dependencies with: "
                "python -m pip install -e ."
            ) from exc

        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection_name(self) -> str:
        return self._collection_name

    def reset(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def existing_ids(self, ids: list[str]) -> set[str]:
        if not ids:
            return set()

        result = self._collection.get(ids=ids)
        return set(result.get("ids", []))

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        if not chunks:
            return

        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=embeddings,
            metadatas=[build_chroma_metadata(chunk) for chunk in chunks],
        )

    def count(self) -> int:
        return self._collection.count()

    def query(
        self,
        *,
        query_embedding: list[float],
        top_k: int,
        document_id: str | None = None,
        file_name: str | None = None,
    ) -> dict:
        where = build_chroma_where(document_id=document_id, file_name=file_name)
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where is not None:
            kwargs["where"] = where

        return self._collection.query(**kwargs)
