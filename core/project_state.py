"""
Project State Tracker
=====================
SQLite-backed persistence layer for the modernization engine.

Tracks:
- File processing status (pending / running / done / failed / skipped)
- File content hashes → skip unchanged files on re-runs
- Per-file transformation audit log (rule, line, before, after, attribution)
- Cross-file symbol registry (class/struct/typedef names and signatures)
- Run history (start, end, statistics)

Usage:
    db = ProjectStateDB("path/to/project.db")
    if db.is_up_to_date("myfile.cpp"):
        return  # skip, already processed and unchanged

    db.mark_running("myfile.cpp")
    ... process ...
    db.mark_done("myfile.cpp", output_path, audit_entries)
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def _file_hash(path: str) -> str:
    """SHA-256 of file content — used to detect unchanged files."""
    try:
        data = Path(path).read_bytes()
        return hashlib.sha256(data).hexdigest()
    except Exception:
        return ""


class ProjectStateDB:
    """
    Persistent SQLite store for multi-file modernization runs.

    Thread-safe: uses WAL mode + connection-per-call pattern.
    """

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS files (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        path        TEXT    UNIQUE NOT NULL,
        hash        TEXT    NOT NULL DEFAULT '',
        status      TEXT    NOT NULL DEFAULT 'pending',
        output_path TEXT,
        run_id      INTEGER,
        complexity  INTEGER DEFAULT 0,
        llm_called  INTEGER DEFAULT 0,
        attribution TEXT,
        duration_ms INTEGER DEFAULT 0,
        processed_at TEXT,
        error_msg   TEXT
    );

    CREATE TABLE IF NOT EXISTS transformations (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id     INTEGER NOT NULL,
        rule        TEXT    NOT NULL,
        line        INTEGER,
        before_text TEXT,
        after_text  TEXT,
        attribution TEXT,
        timestamp   TEXT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS symbols (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        file_id     INTEGER NOT NULL,
        name        TEXT    NOT NULL,
        kind        TEXT,
        signature   TEXT,
        namespace   TEXT
    );

    CREATE TABLE IF NOT EXISTS runs (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at  TEXT NOT NULL,
        finished_at TEXT,
        total_files INTEGER DEFAULT 0,
        passed      INTEGER DEFAULT 0,
        failed      INTEGER DEFAULT 0,
        skipped     INTEGER DEFAULT 0,
        llm_calls   INTEGER DEFAULT 0,
        config_json TEXT,
        submitted_by TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_files_path   ON files(path);
    CREATE INDEX IF NOT EXISTS idx_files_status ON files(status);
    CREATE INDEX IF NOT EXISTS idx_symbols_name ON symbols(name);
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(self._SCHEMA)
            try:
                conn.execute("ALTER TABLE runs ADD COLUMN submitted_by TEXT")
            except sqlite3.OperationalError:
                pass
            conn.commit()
        logger.debug("[ProjectStateDB] Initialized at %s", self.db_path)

    # ── Run management ────────────────────────────────────────────────────

    def start_run(self, total_files: int, config: Optional[dict] = None, submitted_by: Optional[str] = None) -> int:
        """Create a new run record. Returns run_id."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO runs (started_at, total_files, config_json, submitted_by) VALUES (?, ?, ?, ?)",
                (now, total_files, json.dumps(config or {}), submitted_by)
            )
            conn.commit()
            return cur.lastrowid

    def finish_run(self, run_id: int) -> None:
        """Close a run with final statistics."""
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            stats = conn.execute(
                """SELECT
                    COUNT(*) FILTER (WHERE status='done')    AS passed,
                    COUNT(*) FILTER (WHERE status='failed')  AS failed,
                    COUNT(*) FILTER (WHERE status='skipped') AS skipped,
                    SUM(llm_called)                          AS llm_calls
                FROM files WHERE run_id=?""",
                (run_id,)
            ).fetchone()
            conn.execute(
                """UPDATE runs SET finished_at=?, passed=?, failed=?, skipped=?, llm_calls=?
                   WHERE id=?""",
                (now, stats["passed"] or 0, stats["failed"] or 0,
                 stats["skipped"] or 0, stats["llm_calls"] or 0, run_id)
            )
            conn.commit()

    # ── File state management ─────────────────────────────────────────────

    def is_up_to_date(self, file_path: str) -> bool:
        """Return True if this file was successfully processed and is unchanged."""
        current_hash = _file_hash(file_path)
        if not current_hash:
            return False
        with self._connect() as conn:
            row = conn.execute(
                "SELECT hash, status FROM files WHERE path=?", (file_path,)
            ).fetchone()
        if row and row["status"] == "done" and row["hash"] == current_hash:
            logger.debug("[ProjectStateDB] %s is up-to-date, skipping.", file_path)
            return True
        return False

    def register_file(self, file_path: str, run_id: Optional[int] = None) -> int:
        """Register a file for processing. Returns file_id."""
        h = _file_hash(file_path)
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO files (path, hash, status, run_id)
                   VALUES (?, ?, 'pending', ?)
                   ON CONFLICT(path) DO UPDATE SET
                       hash=excluded.hash, status='pending', run_id=excluded.run_id""",
                (file_path, h, run_id)
            )
            conn.commit()
            row = conn.execute("SELECT id FROM files WHERE path=?", (file_path,)).fetchone()
            return row["id"]

    def mark_running(self, file_path: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE files SET status='running' WHERE path=?", (file_path,))
            conn.commit()

    def mark_done(
        self,
        file_path: str,
        output_path: str,
        audit_entries: Optional[List[dict]] = None,
        complexity: int = 0,
        llm_called: bool = False,
        attribution: str = "",
        duration_ms: int = 0,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """UPDATE files SET
                    status='done', output_path=?, processed_at=?,
                    complexity=?, llm_called=?, attribution=?, duration_ms=?
                   WHERE path=?""",
                (output_path, now, complexity, int(llm_called),
                 attribution, duration_ms, file_path)
            )
            if audit_entries:
                file_id = conn.execute(
                    "SELECT id FROM files WHERE path=?", (file_path,)
                ).fetchone()["id"]
                conn.executemany(
                    """INSERT INTO transformations
                       (file_id, rule, line, before_text, after_text, attribution, timestamp)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    [
                        (file_id,
                         e.get("rule", ""),
                         e.get("line"),
                         e.get("before"),
                         e.get("after"),
                         e.get("attribution", attribution),
                         now)
                        for e in audit_entries
                    ]
                )
            conn.commit()

        # Extract and register symbols for header files
        if Path(file_path).suffix.lower() in (".h", ".hpp", ".hxx"):
            try:
                from core.symbol_registry import extract_symbols_from_header
                code = Path(output_path).read_text(encoding="utf-8", errors="replace")
                symbols = extract_symbols_from_header(file_path, code)
                self.register_symbols(file_path, symbols)
                logger.info("Extracted and registered %d symbols for header %s", len(symbols), file_path)
            except Exception as e:
                logger.error("Failed to extract symbols for header %s: %s", file_path, e)

    def mark_failed(self, file_path: str, error: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE files SET status='failed', processed_at=?, error_msg=? WHERE path=?",
                (now, error[:2000], file_path)
            )
            conn.commit()

    def mark_skipped(self, file_path: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE files SET status='skipped', processed_at=? WHERE path=?",
                (now, file_path)
            )
            conn.commit()

    # ── Symbol registry (cross-file type tracking) ────────────────────────

    def register_symbols(self, file_path: str, symbols: List[dict]) -> None:
        """Store public API symbols extracted from a modernized header."""
        with self._connect() as conn:
            file_id = conn.execute(
                "SELECT id FROM files WHERE path=?", (file_path,)
            ).fetchone()
            if not file_id:
                return
            fid = file_id["id"]
            # Clear stale symbols for this file
            conn.execute("DELETE FROM symbols WHERE file_id=?", (fid,))
            conn.executemany(
                "INSERT INTO symbols (file_id, name, kind, signature, namespace) VALUES (?,?,?,?,?)",
                [(fid, s.get("name",""), s.get("kind",""), s.get("signature",""), s.get("namespace",""))
                 for s in symbols]
            )
            conn.commit()

    def get_symbols_for_includes(self, include_names: List[str]) -> List[dict]:
        """Get symbols from files matching the given include names (for cross-file context)."""
        results = []
        with self._connect() as conn:
            for name in include_names:
                # Match by filename stem
                rows = conn.execute(
                    """SELECT s.name, s.kind, s.signature, s.namespace, f.path
                       FROM symbols s JOIN files f ON s.file_id=f.id
                       WHERE f.path LIKE ? AND f.status='done'""",
                    (f"%{name}%",)
                ).fetchall()
                results.extend(dict(r) for r in rows)
        return results

    # ── Reporting ─────────────────────────────────────────────────────────

    def get_run_summary(self, run_id: int) -> dict:
        with self._connect() as conn:
            run = conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
            files = conn.execute(
                "SELECT path, status, complexity, llm_called, duration_ms, attribution, error_msg "
                "FROM files WHERE run_id=?", (run_id,)
            ).fetchall()
        return {
            "run": dict(run) if run else {},
            "files": [dict(f) for f in files],
        }

    def get_runs(self, limit: int = 50) -> List[dict]:
        """Return a list of recent runs ordered by started_at desc."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, started_at, finished_at, total_files, passed, failed, skipped, llm_calls, submitted_by FROM runs ORDER BY started_at DESC LIMIT ?",
                (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_audit_log(self, file_path: str) -> List[dict]:
        with self._connect() as conn:
            fid = conn.execute("SELECT id FROM files WHERE path=?", (file_path,)).fetchone()
            if not fid:
                return []
            rows = conn.execute(
                "SELECT * FROM transformations WHERE file_id=? ORDER BY id",
                (fid["id"],)
            ).fetchall()
        return [dict(r) for r in rows]

    def export_audit_jsonl(self, run_id: int, output_path: str) -> None:
        """Write full audit log for a run as newline-delimited JSON."""
        summary = self.get_run_summary(run_id)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"type": "run_summary", **summary["run"]}) + "\n")
            for file_info in summary["files"]:
                f.write(json.dumps({"type": "file", **file_info}) + "\n")
                audit = self.get_audit_log(file_info["path"])
                for entry in audit:
                    f.write(json.dumps({"type": "transformation", **entry}) + "\n")
        logger.info("[ProjectStateDB] Audit log written to %s", output_path)

    def get_statistics(self) -> dict:
        """Overall project statistics across all runs."""
        with self._connect() as conn:
            stats = conn.execute("""
                SELECT
                    COUNT(*) AS total_files,
                    COUNT(*) FILTER (WHERE status='done')    AS done,
                    COUNT(*) FILTER (WHERE status='failed')  AS failed,
                    COUNT(*) FILTER (WHERE status='skipped') AS skipped,
                    COUNT(*) FILTER (WHERE status='pending') AS pending,
                    AVG(complexity)    AS avg_complexity,
                    SUM(llm_called)    AS total_llm_calls,
                    AVG(duration_ms)   AS avg_duration_ms
                FROM files
            """).fetchone()
        return dict(stats)
