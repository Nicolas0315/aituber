import json
import logging
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MemoryRecord:
    id: int
    text: str
    kind: str
    metadata: Dict[str, Any]
    created_at: str


class MemoryLog:
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
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_memories_kind_id ON memories (kind, id)"
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

    def fetch_by_ids(self, ids: List[int]) -> List[MemoryRecord]:
        if not ids:
            return []
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            placeholders = ",".join("?" for _ in ids)
            rows = conn.execute(
                f"SELECT id, created_at, text, kind, metadata FROM memories WHERE id IN ({placeholders})",
                ids,
            ).fetchall()
            record_map: Dict[int, MemoryRecord] = {}
            for row in rows:
                record_map[row["id"]] = MemoryRecord(
                    id=row["id"],
                    created_at=row["created_at"],
                    text=row["text"],
                    kind=row["kind"],
                    metadata=_parse_metadata(row["metadata"]),
                )
            return [record_map[i] for i in ids if i in record_map]
        finally:
            conn.close()

    def list_since(self, last_id: int, limit: int, kinds: Optional[List[str]] = None) -> List[MemoryRecord]:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            query = "SELECT id, created_at, text, kind, metadata FROM memories WHERE id > ?"
            params: List[Any] = [last_id]
            if kinds:
                placeholders = ",".join("?" for _ in kinds)
                query += f" AND kind IN ({placeholders})"
                params.extend(kinds)
            query += " ORDER BY id ASC LIMIT ?"
            params.append(limit)
            rows = conn.execute(query, params).fetchall()
            return [
                MemoryRecord(
                    id=row["id"],
                    created_at=row["created_at"],
                    text=row["text"],
                    kind=row["kind"],
                    metadata=_parse_metadata(row["metadata"]),
                )
                for row in rows
            ]
        finally:
            conn.close()

    def count_since(self, last_id: int, kinds: Optional[List[str]] = None) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            query = "SELECT COUNT(*) FROM memories WHERE id > ?"
            params: List[Any] = [last_id]
            if kinds:
                placeholders = ",".join("?" for _ in kinds)
                query += f" AND kind IN ({placeholders})"
                params.extend(kinds)
            return int(conn.execute(query, params).fetchone()[0])
        finally:
            conn.close()

    def get_last_summary_marker(self) -> int:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT id, metadata FROM memories WHERE kind = 'summary' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if not row:
                return 0
            metadata = _parse_metadata(row["metadata"])
            if isinstance(metadata, dict) and metadata.get("source_last_id"):
                return int(metadata["source_last_id"])
            return int(row["id"])
        finally:
            conn.close()

    def search_text(self, query: str, limit: int = 5) -> List[MemoryRecord]:
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

            return [
                MemoryRecord(
                    id=row["id"],
                    created_at=row["created_at"],
                    text=row["text"],
                    kind=row["kind"],
                    metadata=_parse_metadata(row["metadata"]),
                )
                for row in rows
            ]
        finally:
            conn.close()


class EmbedderBase:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._dimension: Optional[int] = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            raise RuntimeError("Embedding dimension not initialized yet.")
        return self._dimension

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

    def _set_dimension(self, vector: List[float]) -> None:
        if self._dimension is None:
            self._dimension = len(vector)


class FastEmbedder(EmbedderBase):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        from fastembed import TextEmbedding

        self.model = TextEmbedding(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = [list(vec) for vec in self.model.embed(texts)]
        if embeddings:
            self._set_dimension(embeddings[0])
        return [normalize_vector(vec) for vec in embeddings]


class SentenceTransformerEmbedder(EmbedderBase):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        vectors = [vec.tolist() for vec in embeddings]
        if vectors:
            self._set_dimension(vectors[0])
        return vectors


def normalize_vector(vector: List[float]) -> List[float]:
    norm = sum(v * v for v in vector) ** 0.5
    if norm == 0:
        return vector
    return [v / norm for v in vector]


def create_embedder(model_name: str) -> EmbedderBase:
    errors: List[str] = []
    for provider in ("fastembed", "sentence_transformers"):
        try:
            if provider == "fastembed":
                return FastEmbedder(model_name)
            return SentenceTransformerEmbedder(model_name)
        except Exception as exc:
            errors.append(f"{provider}: {exc}")
    raise RuntimeError("No embedding provider available. Install fastembed or sentence-transformers.")


class VectorStoreBase:
    def add(
        self,
        ids: List[int],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        raise NotImplementedError

    def query(self, embedding: List[float], limit: int) -> List[Tuple[int, float]]:
        raise NotImplementedError


class ChromaVectorStore(VectorStoreBase):
    def __init__(self, path: str, collection_name: str):
        import chromadb

        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(collection_name)

    def add(
        self,
        ids: List[int],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        self.collection.add(
            ids=[str(i) for i in ids],
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, embedding: List[float], limit: int) -> List[Tuple[int, float]]:
        results = self.collection.query(
            query_embeddings=[embedding],
            n_results=limit,
            include=["distances", "ids"],
        )
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]
        return [(int(id_value), float(distance)) for id_value, distance in zip(ids, distances)]


class FaissVectorStore(VectorStoreBase):
    def __init__(self, path: str, dimension: int):
        import faiss
        import numpy as np

        self.faiss = faiss
        self.np = np
        self.path = path
        self.dimension = dimension
        self.index = self._load_or_create()

    def _load_or_create(self):
        if os.path.exists(self.path):
            return self.faiss.read_index(self.path)
        base = self.faiss.IndexFlatIP(self.dimension)
        return self.faiss.IndexIDMap2(base)

    def add(
        self,
        ids: List[int],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        vectors = self.np.array(embeddings, dtype="float32")
        id_array = self.np.array(ids, dtype="int64")
        self.index.add_with_ids(vectors, id_array)
        self.faiss.write_index(self.index, self.path)

    def query(self, embedding: List[float], limit: int) -> List[Tuple[int, float]]:
        vector = self.np.array([embedding], dtype="float32")
        distances, ids = self.index.search(vector, limit)
        return [
            (int(id_value), float(distance))
            for id_value, distance in zip(ids[0], distances[0])
            if id_value >= 0
        ]


class VectorMemoryStore:
    def __init__(
        self,
        log: MemoryLog,
        embedder: Optional[EmbedderBase] = None,
        vector_store: Optional[VectorStoreBase] = None,
    ):
        self.log = log
        self.embedder = embedder
        self.vector_store = vector_store

    def add(self, text: str, kind: str = "short", metadata: Optional[Dict[str, Any]] = None) -> int:
        record_id = self.log.add(text, kind=kind, metadata=metadata)
        if record_id <= 0:
            return record_id
        if self.embedder and self.vector_store:
            try:
                embeddings = self.embedder.embed_documents([text])
                self.vector_store.add(
                    ids=[record_id],
                    embeddings=embeddings,
                    documents=[text],
                    metadatas=[metadata or {}],
                )
            except Exception as exc:
                logger.warning(f"Vector index add failed: {exc}")
        return record_id

    def search(self, query: str, limit: int = 5) -> List[MemoryRecord]:
        if not query:
            return []
        if self.embedder and self.vector_store:
            try:
                embedding = self.embedder.embed_query(query)
                ids = [result[0] for result in self.vector_store.query(embedding, limit)]
                return self.log.fetch_by_ids(ids)
            except Exception as exc:
                logger.warning(f"Vector search failed, falling back to text search: {exc}")
        return self.log.search_text(query, limit)

    def list_since(self, last_id: int, limit: int, kinds: Optional[List[str]] = None) -> List[MemoryRecord]:
        return self.log.list_since(last_id, limit, kinds)

    def count_since(self, last_id: int, kinds: Optional[List[str]] = None) -> int:
        return self.log.count_since(last_id, kinds)

    def get_last_summary_marker(self) -> int:
        return self.log.get_last_summary_marker()


def create_memory_store_from_env() -> VectorMemoryStore:
    db_path = os.getenv("MEMORY_DB", "memory.db")
    log = MemoryLog(db_path=db_path)
    backend = os.getenv("MEMORY_BACKEND", "auto").strip().lower()
    embed_model = os.getenv("MEMORY_EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    vector_dir = os.getenv("MEMORY_VECTOR_DIR", "memory_vectors")
    collection = os.getenv("MEMORY_COLLECTION", "aituber_memory")

    embedder = None
    vector_store: Optional[VectorStoreBase] = None

    if backend in ("auto", "chroma"):
        try:
            os.makedirs(vector_dir, exist_ok=True)
            embedder = create_embedder(embed_model)
            vector_store = ChromaVectorStore(path=vector_dir, collection_name=collection)
            backend = "chroma"
        except Exception as exc:
            logger.warning(f"Chroma backend unavailable: {exc}")
            if backend == "chroma":
                raise

    if vector_store is None and backend in ("auto", "faiss"):
        try:
            os.makedirs(vector_dir, exist_ok=True)
            embedder = create_embedder(embed_model)
            embedder.embed_documents(["dimension probe"])
            vector_path = os.path.join(vector_dir, "memory.faiss")
            vector_store = FaissVectorStore(path=vector_path, dimension=embedder.dimension)
            backend = "faiss"
        except Exception as exc:
            logger.warning(f"FAISS backend unavailable: {exc}")
            if backend == "faiss":
                raise

    if vector_store is None:
        if backend not in ("auto", "sqlite"):
            logger.warning(f"Unknown MEMORY_BACKEND '{backend}', falling back to sqlite")
        embedder = None
        vector_store = None
        backend = "sqlite"

    logger.info(f"Memory backend: {backend}")
    return VectorMemoryStore(log=log, embedder=embedder, vector_store=vector_store)


def _parse_metadata(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {}
