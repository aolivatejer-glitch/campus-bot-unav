import logging
from collections.abc import Iterator, Sequence

from rag_chatbot.chunking.pipeline import read_chunks
from rag_chatbot.config import AppSettings, get_settings
from rag_chatbot.embeddings.base import EmbeddingProvider
from rag_chatbot.embeddings.sentence_transformers_provider import (
    SentenceTransformersEmbeddingProvider,
)
from rag_chatbot.logging_config import LOGGER_NAME
from rag_chatbot.schemas import Chunk, IndexInfo, IndexingResult
from rag_chatbot.vectorstores.base import VectorStore
from rag_chatbot.vectorstores.chroma_store import ChromaVectorStore, is_chromadb_available


def build_index(
    settings: AppSettings | None = None,
    *,
    reset: bool = False,
    limit: int | None = None,
    embedding_provider: EmbeddingProvider | None = None,
    vector_store: VectorStore | None = None,
    logger: logging.Logger | None = None,
) -> IndexingResult:
    settings = settings or get_settings()
    logger = logger or logging.getLogger(LOGGER_NAME)
    settings.ensure_directories()

    if not settings.chunks_file.exists():
        raise FileNotFoundError(
            f"Chunks file not found: {settings.chunks_file}. Run build-chunks first."
        )

    chunks = read_chunks(settings.chunks_file, limit=limit)
    if not chunks:
        raise ValueError(f"No chunks found in {settings.chunks_file}")

    provider = embedding_provider or SentenceTransformersEmbeddingProvider(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    store = vector_store or ChromaVectorStore(
        persist_dir=settings.chroma_dir,
        collection_name=settings.chroma_collection_name,
    )

    if reset:
        store.reset()
        logger.info("Vector collection reset: %s", store.collection_name)

    chunks_indexed = 0
    chunks_skipped = 0
    seen_ids: set[str] = set()

    logger.info("Starting index build for %s chunks", len(chunks))

    for batch in batch_items(chunks, settings.embedding_batch_size):
        unique_batch = [chunk for chunk in batch if chunk.chunk_id not in seen_ids]
        seen_ids.update(chunk.chunk_id for chunk in unique_batch)

        existing_ids = store.existing_ids([chunk.chunk_id for chunk in unique_batch])
        chunks_to_index = [
            chunk for chunk in unique_batch if chunk.chunk_id not in existing_ids
        ]
        chunks_skipped += len(batch) - len(chunks_to_index)

        if not chunks_to_index:
            continue

        embeddings = provider.embed_documents([chunk.text for chunk in chunks_to_index])
        if len(embeddings) != len(chunks_to_index):
            raise ValueError("Embedding provider returned an unexpected number of vectors")

        store.add_chunks(chunks_to_index, embeddings)
        chunks_indexed += len(chunks_to_index)

    logger.info(
        "Index build finished chunks_read=%s indexed=%s skipped=%s collection=%s",
        len(chunks),
        chunks_indexed,
        chunks_skipped,
        store.collection_name,
    )

    return IndexingResult(
        chunks_read=len(chunks),
        chunks_indexed=chunks_indexed,
        chunks_skipped=chunks_skipped,
        collection_name=store.collection_name,
        chroma_dir=str(settings.chroma_dir),
        embedding_model=provider.model_name,
        embedding_device=provider.device,
    )


def get_index_info(
    settings: AppSettings | None = None,
    *,
    vector_store: VectorStore | None = None,
) -> IndexInfo:
    settings = settings or get_settings()
    chroma_available = is_chromadb_available()
    vector_count: int | None = None
    error_message: str | None = None

    if vector_store is not None:
        vector_count = vector_store.count()
        chroma_available = True
    elif chroma_available:
        try:
            vector_count = ChromaVectorStore(
                persist_dir=settings.chroma_dir,
                collection_name=settings.chroma_collection_name,
            ).count()
        except Exception as exc:
            error_message = str(exc)

    return IndexInfo(
        collection_name=settings.chroma_collection_name,
        chroma_dir=str(settings.chroma_dir),
        embedding_model=settings.embedding_model,
        embedding_device=settings.embedding_device,
        chroma_available=chroma_available,
        vector_count=vector_count,
        error_message=error_message,
    )


def batch_items(items: Sequence[Chunk], batch_size: int) -> Iterator[list[Chunk]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    for start in range(0, len(items), batch_size):
        yield list(items[start : start + batch_size])
