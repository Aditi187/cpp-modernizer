import os
import hashlib
import logging
from typing import Dict, Optional, Any
from collections import OrderedDict
from pathlib import Path

from dataclasses import dataclass

try:
    import agents.workflow.config as config_mod
    WorkflowConfig = config_mod.WorkflowConfig
except (ImportError, AttributeError):
    raise ImportError("WorkflowConfig could not be imported from agents.workflow.config")

logger = logging.getLogger(__name__)


@dataclass
class WorkflowContext:
    """Persistent runtime context shared across workflow nodes.

    Phase 4 additions:
        - multi-model caching
        - semantic cache
        - dependency graph reuse
        - planner history
        - transformation stats
    """
    config: Any = None
    db_path: str = os.environ.get("CACHE_DB_PATH", str(Path(__file__).parent.parent.parent / ".modernization_cache.db"))
    code_graph: Any = None
    semantic_cache: Dict[str, Dict[str, Any]] = None
    planner_history: list[Any] = None
    transformation_stats: Dict[str, int] = None
    session_id: str = ""
    total_tokens: int = 0
    llm_calls_succeeded: int = 0
    _conn: Any = None

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = WorkflowConfig.from_env()
        if self.semantic_cache is None:
            self.semantic_cache = {}
        if self.planner_history is None:
            self.planner_history = []
        if self.transformation_stats is None:
            self.transformation_stats = {}
        if not self.session_id:
            self.session_id = self._generate_session_id()

        import sqlite3
        self._conn = sqlite3.connect(self.db_path, timeout=30.0, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")

        # Initialize SQLite database
        self._init_db()

        logger.info(f"[WorkflowContext] session initialized | id={self.session_id}")

    def __del__(self) -> None:
        if hasattr(self, "_conn") and self._conn:
            try:
                self._conn.close()
            except Exception:
                pass

    def _init_db(self) -> None:
        try:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS llm_cache (
                    key TEXT PRIMARY KEY,
                    response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            self._conn.commit()
        except Exception as e:
            logger.error(f"Failed to initialize SQLite cache database: {e}")

    # --- Session Helpers ---

    def _generate_session_id(self) -> str:
        random_bytes = os.urandom(8)
        return hashlib.sha1(random_bytes).hexdigest()[:10]

    # --- LLM Response Cache ---

    def get_cached_llm_response(self, cache_key: str) -> Optional[str]:
        if not cache_key:
            return None
        from datetime import datetime, timedelta, timezone
        key = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        ttl_seconds = int(os.environ.get("CACHE_TTL_SECONDS", 604800))
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT response, created_at FROM llm_cache WHERE key = ?", (key,))
            row = cursor.fetchone()
            if row:
                response, created_at_str = row
                # SQLite CURRENT_TIMESTAMP format can vary; try multiple formats
                for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S.%f"):
                    try:
                        created_at = datetime.strptime(created_at_str, fmt).replace(tzinfo=timezone.utc)
                        break
                    except ValueError:
                        continue
                else:
                    # Fallback: treat as expired so we re-fetch
                    created_at = datetime.min.replace(tzinfo=timezone.utc)
                if datetime.now(timezone.utc) - created_at < timedelta(seconds=ttl_seconds):
                    return response
                else:
                    cursor.execute("DELETE FROM llm_cache WHERE key = ?", (key,))
                    self._conn.commit()
        except Exception as e:
            logger.error(f"Error reading from SQLite cache: {e}")
        return None

    def cache_llm_response(self, cache_key: str, response: str) -> None:
        if not cache_key or not response:
            return
        key = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
        try:
            self._conn.execute(
                "INSERT OR REPLACE INTO llm_cache (key, response) VALUES (?, ?)",
                (key, response)
            )
            self._conn.commit()
        except Exception as e:
            logger.error(f"Error writing to SQLite cache: {e}")

    # --- Semantic Cache ---

    def cache_semantic_result(self, code_hash: str, result: Dict[str, Any]) -> None:
        self.semantic_cache[code_hash] = result

    def get_semantic_result(self, code_hash: str) -> Optional[Dict[str, Any]]:
        return self.semantic_cache.get(code_hash)

    # --- Transformation Stats ---

    def record_transformation(self, rule_name: str):
        self.transformation_stats[rule_name] = (
            self.transformation_stats.get(rule_name, 0) + 1
        )

    def add_tokens(self, count: int) -> None:
        """Accumulate token usage from an LLM call."""
        if count and count > 0:
            self.total_tokens += count

    def _get_cache_size(self) -> int:
        try:
            cursor = self._conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM llm_cache")
            return cursor.fetchone()[0]
        except Exception:
            return 0

    # --- Serialization ---

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "model": self.config.model_name,
            "cache_entries": self._get_cache_size(),
            "transformations": len(self.transformation_stats),
        }

    def __repr__(self) -> str:
        return (
            "WorkflowContext("
            f"session_id={self.session_id}, "
            f"model={self.config.model_name}, "
            f"cache={self._get_cache_size()}"
            ")"
        )