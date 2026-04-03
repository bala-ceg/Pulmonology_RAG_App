"""
Database service — PostgreSQL connection pool with per-request fallback.

Provides:
- ``init_pool()``       — call once at app startup to open the connection pool.
- ``get_connection()``  — context manager yielding a psycopg connection.
- ``param_placeholder`` — ``%s`` (psycopg) or ``?`` (SQLite).
- ``execute_query()``   — runs a query, adapting placeholders automatically.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Generator

import psycopg

from config import Config
from utils.error_handlers import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Internal state
# ---------------------------------------------------------------------------
_pg_pool: Any = None  # psycopg_pool.ConnectionPool | None
_pool_available: bool = False

# ---------------------------------------------------------------------------
# Param-placeholder
# ---------------------------------------------------------------------------

#: SQL parameter placeholder.  Always ``%s`` when a real PostgreSQL
#: connection is in use.  Exposed so callers can swap ``?`` → ``%s`` when
#: adapting SQLite queries.
param_placeholder: str = "%s"


# ---------------------------------------------------------------------------
# Pool initialisation
# ---------------------------------------------------------------------------

def init_pool() -> None:
    """Open the psycopg connection pool (non-blocking).

    Falls back silently to per-request connections if ``psycopg_pool`` is not
    installed.  Safe to call multiple times — subsequent calls are no-ops.
    """
    global _pg_pool, _pool_available

    if _pool_available:
        return  # Already initialised

    try:
        from psycopg_pool import ConnectionPool as _PGPool  # type: ignore[import]

        pool_kwargs = Config.db_kwargs()
        _pg_pool = _PGPool(
            conninfo="",
            kwargs=pool_kwargs,
            min_size=Config.DB_POOL_MIN_SIZE,
            max_size=Config.DB_POOL_MAX_SIZE,
            open=False,
            reconnect_timeout=Config.DB_POOL_RECONNECT_TIMEOUT,
        )
        _pg_pool.open(wait=False)  # Background open — does not block startup
        _pool_available = True
        logger.info("PostgreSQL connection pool initialised (psycopg_pool)")
    except ImportError:
        _pg_pool = None
        _pool_available = False
        logger.warning(
            "psycopg_pool not installed — using per-request connections. "
            "Run: pip install 'psycopg[pool]'"
        )


# ---------------------------------------------------------------------------
# Connection context manager
# ---------------------------------------------------------------------------

@contextmanager
def get_connection() -> Generator[psycopg.Connection, None, None]:
    """Yield a psycopg connection from the pool or as a fresh connection.

    Usage::

        with get_connection() as conn:
            conn.execute("SELECT 1")

    When the pool is available connections are returned to the pool on exit.
    Otherwise a new connection is opened and closed per call.
    """
    if _pool_available and _pg_pool is not None:
        with _pg_pool.connection() as conn:
            yield conn
    else:
        with psycopg.connect(**Config.db_kwargs()) as conn:
            yield conn


# ---------------------------------------------------------------------------
# Query helper
# ---------------------------------------------------------------------------

def execute_query(
    conn: psycopg.Connection,
    query: str,
    params: tuple[Any, ...] | list[Any] | None = None,
) -> psycopg.Cursor:
    """Execute *query* on *conn*, adapting ``?`` placeholders to ``%s``.

    This helper lets callers write SQLite-style ``?`` placeholders (common
    throughout the codebase for the SQLite fallback path) and have them
    transparently rewritten to ``%s`` when running against PostgreSQL.

    Returns the cursor so callers can call ``.fetchone()`` / ``.fetchall()``.
    """
    adapted = query.replace("?", "%s")
    cursor = conn.cursor()
    if params:
        cursor.execute(adapted, params)
    else:
        cursor.execute(adapted)
    return cursor


# ---------------------------------------------------------------------------
# Initialise pool at import time so the app doesn't need an explicit call
# ---------------------------------------------------------------------------
init_pool()
