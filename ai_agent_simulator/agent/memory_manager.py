"""SQLite-backed memory for agent interactions.

The :class:`MemoryManager` encapsulates all persistence concerns for the
simulator: schema creation, inserting new interactions, and replaying a
user's history. The database file is created on demand, so callers do
not need to provision anything ahead of time.

Schema
------

Table ``interactions``:

================  ==========  =======================================
Column            Type        Notes
================  ==========  =======================================
``id``            INTEGER     Primary key, auto-increment.
``user_id``       TEXT        Opaque identifier for the end user.
``message``       TEXT        The raw user message.
``intent``        TEXT        The intent label predicted by the agent.
``timestamp``     DATETIME    UTC timestamp, defaults to ``CURRENT_TIMESTAMP``.
================  ==========  =======================================
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, List, Optional, Union

DEFAULT_DB_PATH: Path = Path("agent_memory.db")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS interactions (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id   TEXT    NOT NULL,
    message   TEXT    NOT NULL,
    intent    TEXT    NOT NULL,
    timestamp DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_interactions_user_time
    ON interactions (user_id, timestamp DESC);
"""


@dataclass(frozen=True)
class Interaction:
    """A single stored interaction between a user and the agent."""

    id: int
    user_id: str
    message: str
    intent: str
    timestamp: datetime

    @classmethod
    def from_row(cls, row: sqlite3.Row) -> "Interaction":
        """Build an :class:`Interaction` from a ``sqlite3.Row``."""
        raw_ts = row["timestamp"]
        if isinstance(raw_ts, datetime):
            ts = raw_ts
        else:
            ts = _parse_sqlite_timestamp(str(raw_ts))
        return cls(
            id=int(row["id"]),
            user_id=str(row["user_id"]),
            message=str(row["message"]),
            intent=str(row["intent"]),
            timestamp=ts,
        )


class MemoryManager:
    """Persistence layer for agent interactions.

    The manager opens a new short-lived connection for every operation.
    SQLite handles this gracefully for single-process usage and it keeps
    the API thread-safe without extra locking.
    """

    def __init__(self, db_path: Union[str, Path] = DEFAULT_DB_PATH) -> None:
        """Create a manager bound to ``db_path``.

        The database file and schema are created lazily the first time
        :meth:`initialize_db` (or any write method) is invoked.

        Args:
            db_path: Filesystem path to the SQLite database file.
        """
        self._db_path = Path(db_path)
        self._initialized = False

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------
    def initialize_db(self) -> None:
        """Create the database file and schema if they don't exist."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE_SQL)
            conn.execute(_CREATE_INDEX_SQL)
            conn.commit()
        self._initialized = True

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------
    def store_interaction(self, user_id: str, message: str, intent: str) -> int:
        """Persist a single interaction and return its row id.

        Args:
            user_id: Opaque identifier for the end user.
            message: The raw user message.
            intent: Predicted intent label.

        Returns:
            The auto-incremented primary key of the new row.

        Raises:
            ValueError: If ``user_id``, ``message``, or ``intent`` is
                empty or blank.
        """
        if not user_id or not user_id.strip():
            raise ValueError("user_id must be a non-empty string")
        if not message or not message.strip():
            raise ValueError("message must be a non-empty string")
        if not intent or not intent.strip():
            raise ValueError("intent must be a non-empty string")

        self._ensure_initialized()
        with self._connect() as conn:
            insert_result = conn.execute(
                """
                INSERT INTO interactions (user_id, message, intent)
                VALUES (?, ?, ?)
                """,
                (user_id, message, intent),
            )
            conn.commit()
            return int(insert_result.lastrowid)

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def get_user_history(
        self, user_id: str, limit: Optional[int] = None
    ) -> List[Interaction]:
        """Return all interactions for ``user_id`` ordered oldest-first.

        Args:
            user_id: Opaque identifier for the end user.
            limit: Optional cap on the number of rows returned. When
                provided, the *most recent* ``limit`` interactions are
                returned (still oldest-first within that window).

        Returns:
            A list of :class:`Interaction` objects. Empty if the user
            has no history.
        """
        if not user_id or not user_id.strip():
            return []

        self._ensure_initialized()
        with self._connect() as conn:
            if limit is None:
                rows = conn.execute(
                    """
                    SELECT id, user_id, message, intent, timestamp
                    FROM interactions
                    WHERE user_id = ?
                    ORDER BY timestamp ASC, id ASC
                    """,
                    (user_id,),
                ).fetchall()
            else:
                if limit <= 0:
                    return []
                rows = conn.execute(
                    """
                    SELECT id, user_id, message, intent, timestamp FROM (
                        SELECT id, user_id, message, intent, timestamp
                        FROM interactions
                        WHERE user_id = ?
                        ORDER BY timestamp DESC, id DESC
                        LIMIT ?
                    ) ORDER BY timestamp ASC, id ASC
                    """,
                    (user_id, int(limit)),
                ).fetchall()
        return [Interaction.from_row(row) for row in rows]

    def get_last_intent(self, user_id: str) -> Optional[str]:
        """Return the most recent intent for ``user_id``, or ``None``.

        Args:
            user_id: Opaque identifier for the end user.

        Returns:
            The intent label from the newest interaction, or ``None``
            when the user has no history.
        """
        if not user_id or not user_id.strip():
            return None

        self._ensure_initialized()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT intent
                FROM interactions
                WHERE user_id = ?
                ORDER BY timestamp DESC, id DESC
                LIMIT 1
                """,
                (user_id,),
            ).fetchone()
        return str(row["intent"]) if row is not None else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _ensure_initialized(self) -> None:
        """Run :meth:`initialize_db` the first time it's needed."""
        if not self._initialized:
            self.initialize_db()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a short-lived SQLite connection with row access by name."""
        conn = sqlite3.connect(
            str(self._db_path),
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
        )
        try:
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            conn.close()


def _parse_sqlite_timestamp(raw: str) -> datetime:
    """Parse the timestamp string format emitted by SQLite.

    SQLite's ``CURRENT_TIMESTAMP`` produces ``YYYY-MM-DD HH:MM:SS``.
    ``PARSE_DECLTYPES`` can deliver either a string or a datetime
    depending on how the column is declared, so we handle both.
    """
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    # Fall back to fromisoformat which handles ``T`` separators etc.
    return datetime.fromisoformat(raw)
