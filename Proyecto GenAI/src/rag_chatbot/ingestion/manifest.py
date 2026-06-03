import sqlite3
from pathlib import Path

from rag_chatbot.schemas import ManifestRecord


class DocumentManifest:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY,
                    file_name TEXT NOT NULL,
                    file_path TEXT NOT NULL UNIQUE,
                    file_type TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    requires_ocr INTEGER NOT NULL,
                    page_count INTEGER NOT NULL,
                    char_count INTEGER NOT NULL,
                    processed_at TEXT NOT NULL,
                    error_message TEXT
                )
                """
            )
            connection.commit()

    def has_same_hash(self, file_path: Path, file_hash: str) -> bool:
        record = self.get_by_path(file_path)
        return record is not None and record.file_hash == file_hash

    def get_by_path(self, file_path: Path) -> ManifestRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE file_path = ?",
                (str(file_path.resolve()),),
            ).fetchone()

        return self._row_to_record(row) if row else None

    def get(self, document_id: str) -> ManifestRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM documents WHERE document_id = ?",
                (document_id,),
            ).fetchone()

        return self._row_to_record(row) if row else None

    def upsert(self, record: ManifestRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents (
                    document_id,
                    file_name,
                    file_path,
                    file_type,
                    file_hash,
                    status,
                    requires_ocr,
                    page_count,
                    char_count,
                    processed_at,
                    error_message
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(document_id) DO UPDATE SET
                    file_name = excluded.file_name,
                    file_path = excluded.file_path,
                    file_type = excluded.file_type,
                    file_hash = excluded.file_hash,
                    status = excluded.status,
                    requires_ocr = excluded.requires_ocr,
                    page_count = excluded.page_count,
                    char_count = excluded.char_count,
                    processed_at = excluded.processed_at,
                    error_message = excluded.error_message
                """,
                (
                    record.document_id,
                    record.file_name,
                    record.file_path,
                    record.file_type,
                    record.file_hash,
                    record.status.value,
                    int(record.requires_ocr),
                    record.page_count,
                    record.char_count,
                    record.processed_at.isoformat(),
                    record.error_message,
                ),
            )
            connection.commit()

    def list_documents(self) -> list[ManifestRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM documents ORDER BY processed_at DESC, file_name ASC"
            ).fetchall()

        return [self._row_to_record(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ManifestRecord:
        return ManifestRecord(
            document_id=row["document_id"],
            file_name=row["file_name"],
            file_path=row["file_path"],
            file_type=row["file_type"],
            file_hash=row["file_hash"],
            status=row["status"],
            requires_ocr=bool(row["requires_ocr"]),
            page_count=row["page_count"],
            char_count=row["char_count"],
            processed_at=row["processed_at"],
            error_message=row["error_message"],
        )
