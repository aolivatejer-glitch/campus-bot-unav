import json

import pytest

from rag_chatbot.config import AppSettings
from rag_chatbot.embeddings.base import EmbeddingProvider
from rag_chatbot.indexing.pipeline import batch_items, build_index, get_index_info
from rag_chatbot.schemas import Chunk
from rag_chatbot.vectorstores.base import VectorStore


class FakeEmbeddingProvider(EmbeddingProvider):
    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def device(self) -> str:
        return "cpu"

    @property
    def dimension(self) -> int:
        return 3

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(index), float(len(text)), 1.0] for index, text in enumerate(texts)]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 0.0, 1.0]


class FakeVectorStore(VectorStore):
    def __init__(self, existing_ids: set[str] | None = None) -> None:
        self.ids = set(existing_ids or set())
        self.added_chunks: list[Chunk] = []
        self.reset_called = False

    @property
    def collection_name(self) -> str:
        return "documents"

    def reset(self) -> None:
        self.reset_called = True
        self.ids.clear()

    def existing_ids(self, ids: list[str]) -> set[str]:
        return self.ids.intersection(ids)

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        assert len(chunks) == len(embeddings)
        self.added_chunks.extend(chunks)
        self.ids.update(chunk.chunk_id for chunk in chunks)

    def count(self) -> int:
        return len(self.ids)


def _settings(tmp_path) -> AppSettings:
    return AppSettings(
        documents_dir=tmp_path / "Documentos",
        data_dir=tmp_path / "data",
        processed_dir=tmp_path / "data" / "processed",
        eval_dir=tmp_path / "data" / "eval",
        chunks_dir=tmp_path / "data" / "chunks",
        chunks_file=tmp_path / "data" / "chunks" / "chunks.jsonl",
        storage_dir=tmp_path / "storage",
        chroma_dir=tmp_path / "storage" / "chroma",
        manifest_db_path=tmp_path / "storage" / "manifest.sqlite",
        log_dir=tmp_path / "logs",
        embedding_batch_size=2,
        _env_file=None,
    )


def _chunk(chunk_id: str, text: str = "Texto del chunk") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id="doc_1",
        file_name="sample.pdf",
        file_path="Documentos/sample.pdf",
        file_type="pdf",
        page_number=1,
        chunk_index=0,
        text=text,
        char_count=len(text),
        metadata={"source": "Documentos/sample.pdf"},
    )


def _toc_chunk(chunk_id: str) -> Chunk:
    chunk = _chunk(chunk_id, "ÍNDICE 1. Introducción ................ 2")
    chunk.metadata["is_toc_candidate"] = True
    chunk.metadata["text_quality"] = "low"
    return chunk


def _write_chunks(settings: AppSettings, chunks: list[Chunk]) -> None:
    settings.chunks_file.parent.mkdir(parents=True, exist_ok=True)
    with settings.chunks_file.open("w", encoding="utf-8") as file:
        for chunk in chunks:
            file.write(json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) + "\n")


def test_batch_items_groups_chunks() -> None:
    chunks = [_chunk("c1"), _chunk("c2"), _chunk("c3")]

    batches = list(batch_items(chunks, batch_size=2))

    assert [[chunk.chunk_id for chunk in batch] for batch in batches] == [
        ["c1", "c2"],
        ["c3"],
    ]


def test_build_index_reads_chunks_and_indexes_with_fake_provider(tmp_path) -> None:
    settings = _settings(tmp_path)
    chunks = [_chunk("c1"), _chunk("c2")]
    _write_chunks(settings, chunks)
    store = FakeVectorStore()

    result = build_index(
        settings,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
    )

    assert result.chunks_read == 2
    assert result.chunks_indexed == 2
    assert result.chunks_skipped == 0
    assert [chunk.chunk_id for chunk in store.added_chunks] == ["c1", "c2"]


def test_build_index_skips_existing_ids(tmp_path) -> None:
    settings = _settings(tmp_path)
    _write_chunks(settings, [_chunk("c1"), _chunk("c2")])
    store = FakeVectorStore(existing_ids={"c1"})

    result = build_index(
        settings,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
    )

    assert result.chunks_read == 2
    assert result.chunks_indexed == 1
    assert result.chunks_skipped == 1
    assert [chunk.chunk_id for chunk in store.added_chunks] == ["c2"]


def test_build_index_skips_toc_candidate_chunks(tmp_path) -> None:
    settings = _settings(tmp_path)
    _write_chunks(settings, [_toc_chunk("toc"), _chunk("c1")])
    store = FakeVectorStore()

    result = build_index(
        settings,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
    )

    assert result.chunks_read == 2
    assert result.chunks_indexed == 1
    assert result.chunks_skipped == 1
    assert [chunk.chunk_id for chunk in store.added_chunks] == ["c1"]


def test_build_index_reset_reindexes_existing_ids(tmp_path) -> None:
    settings = _settings(tmp_path)
    _write_chunks(settings, [_chunk("c1"), _chunk("c2")])
    store = FakeVectorStore(existing_ids={"c1", "c2"})

    result = build_index(
        settings,
        reset=True,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
    )

    assert store.reset_called is True
    assert result.chunks_indexed == 2
    assert result.chunks_skipped == 0


def test_build_index_limit_indexes_only_first_chunks(tmp_path) -> None:
    settings = _settings(tmp_path)
    _write_chunks(settings, [_chunk("c1"), _chunk("c2"), _chunk("c3")])
    store = FakeVectorStore()

    result = build_index(
        settings,
        limit=2,
        embedding_provider=FakeEmbeddingProvider(),
        vector_store=store,
    )

    assert result.chunks_read == 2
    assert [chunk.chunk_id for chunk in store.added_chunks] == ["c1", "c2"]


def test_build_index_errors_when_chunks_file_missing(tmp_path) -> None:
    settings = _settings(tmp_path)

    with pytest.raises(FileNotFoundError, match="Run build-chunks first"):
        build_index(
            settings,
            embedding_provider=FakeEmbeddingProvider(),
            vector_store=FakeVectorStore(),
        )


def test_get_index_info_with_fake_store(tmp_path) -> None:
    settings = _settings(tmp_path)
    store = FakeVectorStore(existing_ids={"c1", "c2"})

    info = get_index_info(settings, vector_store=store)

    assert info.chroma_available is True
    assert info.vector_count == 2
    assert info.collection_name == "documents"
