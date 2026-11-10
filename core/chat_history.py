"""SQLite-backed conversation history. Stores messages and source references only
(never API keys or full document text)."""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.settings import HISTORY_DB


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class ChatHistory:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = Path(db_path or HISTORY_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY, title TEXT, document_names TEXT,
                    created_at TEXT, updated_at TEXT);
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL, content TEXT NOT NULL,
                    sources TEXT, meta TEXT, created_at TEXT,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE);
                """
            )

    def _conn(self):
        conn = sqlite3.connect(str(self.db_path), timeout=10)
        conn.execute("PRAGMA foreign_keys = ON")
        return _Conn(conn)

    def create_conversation(self, document_names: Optional[List[str]] = None, title: str = "New chat") -> str:
        cid = uuid.uuid4().hex[:12]
        now = _now()
        with self._conn() as c:
            c.execute("INSERT INTO conversations VALUES (?,?,?,?,?)",
                      (cid, title, json.dumps(document_names or []), now, now))
        return cid

    def add_message(self, conversation_id: str, role: str, content: str,
                    sources: Optional[List[Dict[str, Any]]] = None,
                    meta: Optional[Dict[str, Any]] = None) -> None:
        slim = [{k: s.get(k) for k in ("source", "page", "chunk_id", "score")} for s in (sources or [])]
        with self._conn() as c:
            c.execute("INSERT INTO messages (conversation_id, role, content, sources, meta, created_at) "
                      "VALUES (?,?,?,?,?,?)",
                      (conversation_id, role, content, json.dumps(slim), json.dumps(meta or {}), _now()))
            if role == "user":
                row = c.execute("SELECT title FROM conversations WHERE id=?", (conversation_id,)).fetchone()
                if row and row[0] == "New chat":
                    c.execute("UPDATE conversations SET title=? WHERE id=?",
                              (content.strip().replace("\n", " ")[:60] or "New chat", conversation_id))
            c.execute("UPDATE conversations SET updated_at=? WHERE id=?", (_now(), conversation_id))

    def get_messages(self, conversation_id: str) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute("SELECT role, content, sources, meta, created_at FROM messages "
                             "WHERE conversation_id=? ORDER BY id", (conversation_id,)).fetchall()
        return [{"role": r[0], "content": r[1], "sources": json.loads(r[2] or "[]"),
                 "meta": json.loads(r[3] or "{}"), "created_at": r[4]} for r in rows]

    def list_conversations(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT c.id, c.title, c.document_names, c.created_at, c.updated_at, "
                "(SELECT COUNT(*) FROM messages m WHERE m.conversation_id=c.id) "
                "FROM conversations c ORDER BY c.updated_at DESC, c.rowid DESC LIMIT ?", (limit,)).fetchall()
        return [{"conversation_id": r[0], "title": r[1], "document_names": json.loads(r[2] or "[]"),
                 "created_at": r[3], "updated_at": r[4], "message_count": r[5]} for r in rows]

    def delete_conversation(self, conversation_id: str) -> None:
        with self._conn() as c:
            c.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
            c.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))

    def export_markdown(self, conversation_id: str) -> str:
        lines = ["# IntelliAssist AI conversation", ""]
        for m in self.get_messages(conversation_id):
            who = "You" if m["role"] == "user" else "IntelliAssist AI"
            lines += [f"**{who}** ({m['created_at']})", "", m["content"], ""]
            for s in m["sources"]:
                lines.append(f"> Source: {s['source']}" + (f" — Page {s['page']}" if s.get("page") else ""))
            lines.append("")
        return "\n".join(lines)


class _Conn:
    """sqlite3 connection that commits and *closes* on exit (the stdlib one only commits)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self.conn = conn

    def __enter__(self) -> sqlite3.Connection:
        return self.conn

    def __exit__(self, exc_type, exc, tb) -> None:
        with closing(self.conn):
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
