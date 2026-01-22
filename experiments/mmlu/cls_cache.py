from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _chunked(seq: Sequence[str], chunk_size: int) -> Iterable[List[str]]:
    chunk_size = max(1, int(chunk_size))
    for i in range(0, len(seq), chunk_size):
        yield list(seq[i : i + chunk_size])


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    puts: int


class SqliteClsCache:
    """
    A tiny SQLite-backed cache for frozen CLS embeddings.

    - Key: str
    - Value: np.ndarray[float16] of shape [dim]

    Notes
    -----
    - Uses WAL mode for safer concurrent reads.
    - Intended for single-process training; multi-process writes may contend.
    """

    def __init__(
        self,
        path: str,
        *,
        dim: int,
        dtype: str = "f16",
        timeout_s: int = 30,
    ) -> None:
        self.path = os.path.abspath(path)
        self.dim = int(dim)
        self.dtype = str(dtype)
        if self.dim <= 0:
            raise ValueError("SqliteClsCache: dim must be > 0")
        if self.dtype not in {"f16", "f32"}:
            raise ValueError("SqliteClsCache: dtype must be 'f16' or 'f32'")

        _ensure_parent_dir(self.path)
        self._conn = sqlite3.connect(self.path, timeout=float(timeout_s))
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA temp_store=MEMORY;")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cls (
                k TEXT PRIMARY KEY,
                dim INTEGER NOT NULL,
                dtype TEXT NOT NULL,
                data BLOB NOT NULL
            );
            """
        )
        self._conn.execute("CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);")
        self._conn.execute("INSERT OR REPLACE INTO meta(k, v) VALUES ('dim', ?);", (str(self.dim),))
        self._conn.execute("INSERT OR REPLACE INTO meta(k, v) VALUES ('dtype', ?);", (str(self.dtype),))
        self._conn.commit()

        self._hits = 0
        self._misses = 0
        self._puts = 0

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    def stats(self) -> CacheStats:
        return CacheStats(hits=int(self._hits), misses=int(self._misses), puts=int(self._puts))

    def reset_stats(self) -> None:
        self._hits = 0
        self._misses = 0
        self._puts = 0

    def get_many(self, keys: Sequence[str], *, chunk_size: int = 900) -> Dict[str, np.ndarray]:
        if not keys:
            return {}
        found: Dict[str, np.ndarray] = {}
        for chunk in _chunked(list(keys), chunk_size=int(chunk_size)):
            placeholders = ",".join(["?"] * len(chunk))
            cur = self._conn.execute(f"SELECT k, dim, dtype, data FROM cls WHERE k IN ({placeholders});", tuple(chunk))
            rows = cur.fetchall()
            for k, dim, dtype, data in rows:
                if int(dim) != self.dim:
                    continue
                if dtype == "f16":
                    arr = np.frombuffer(data, dtype=np.float16)
                    if arr.size != self.dim:
                        continue
                    found[str(k)] = arr.astype(np.float32, copy=False)
                elif dtype == "f32":
                    arr = np.frombuffer(data, dtype=np.float32)
                    if arr.size != self.dim:
                        continue
                    found[str(k)] = arr.astype(np.float32, copy=False)

        self._hits += len(found)
        self._misses += int(len(keys) - len(found))
        return found

    def put_many(self, items: Sequence[Tuple[str, np.ndarray]]) -> None:
        rows = []
        dtype = self.dtype
        for k, v in items:
            if v is None:
                continue
            arr = np.asarray(v, dtype=np.float32).reshape(-1)
            if arr.size != self.dim:
                raise ValueError(f"SqliteClsCache: embedding dim mismatch for key={k} (got {arr.size}, want {self.dim})")
            if dtype == "f16":
                payload = arr.astype(np.float16, copy=False).tobytes()
            else:
                payload = arr.astype(np.float32, copy=False).tobytes()
            rows.append((str(k), int(self.dim), str(dtype), payload))

        if not rows:
            return

        with self._conn:
            self._conn.executemany("INSERT OR REPLACE INTO cls(k, dim, dtype, data) VALUES (?, ?, ?, ?);", rows)
        self._puts += len(rows)

