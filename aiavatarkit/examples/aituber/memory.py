import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class MemoryRecord:
    id: int
    text: str
    kind: str
    metadata: Dict[str, Any]
    created_at: str


class SimpleMemoryStore:
    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self.has_fts = False
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS memories (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        created_at TEXT NOT NULL,
                        text TEXT NOT NULL,
                        kind TEXT NOT NULL,
                        metadata TEXT
                    )
                    """
                )
                try:
                    conn.execute(
                        """
                        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                        USING fts5(text, content='memories', content_rowid='id')
                        """
                    )
                    self.has_fts = True
                except sqlite3.OperationalError:
                    self.has_fts = False
        finally:
            conn.close()

    def add(self, text: str, kind: str = "short", metadata: Optional[Dict[str, Any]] = None) -> int:
        if not text:
            return -1

        payload = json.dumps(metadata or {}, ensure_ascii=True)
        created_at = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        try:
            with conn:
                cursor = conn.execute(
                    "INSERT INTO memories (created_at, text, kind, metadata) VALUES (?, ?, ?, ?)",
                    (created_at, text, kind, payload),
                )
                row_id = cursor.lastrowid
                if self.has_fts:
                    conn.execute(
                        "INSERT INTO memories_fts (rowid, text) VALUES (?, ?)",
                        (row_id, text),
                    )
                return row_id
        finally:
            conn.close()

    def search(self, query: str, limit: int = 5) -> List[MemoryRecord]:
        if not query:
            return []

        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            if self.has_fts:
                rows = conn.execute(
                    """
                    SELECT m.id, m.created_at, m.text, m.kind, m.metadata
                    FROM memories_fts f
                    JOIN memories m ON m.id = f.rowid
                    WHERE memories_fts MATCH ?
                    ORDER BY m.id DESC
                    LIMIT ?
                    """,
                    (query, limit),
                ).fetchall()
            else:
                like_query = f"%{query}%"
                rows = conn.execute(
                    """
                    SELECT id, created_at, text, kind, metadata
                    FROM memories
                    WHERE text LIKE ?
                    ORDER BY id DESC
                    LIMIT ?
                    """,
                    (like_query, limit),
                ).fetchall()

            results: List[MemoryRecord] = []
            for row in rows:
                metadata = {}
                try:
                    metadata = json.loads(row["metadata"] or "{}")
                except json.JSONDecodeError:
                    metadata = {}
                results.append(
                    MemoryRecord(
                        id=row["id"],
                        created_at=row["created_at"],
                        text=row["text"],
                        kind=row["kind"],
                        metadata=metadata,
                    )
                )
            return results
        finally:
            conn.close()
