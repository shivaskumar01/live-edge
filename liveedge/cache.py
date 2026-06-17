"""Local parquet cache for the slow / rate-limited historical pulls.

Cached frames live under ``LIVEEDGE_CACHE_DIR`` (default ``./.cache``) and are gitignored. Only
successful, non-empty frames are cached, so a failed or rate-limited pull is simply retried on
the next run, which is what makes a full-season nba_api pull resumable rather than all-or-
nothing. Delete the cache directory to force a fresh download.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pandas as pd


def cache_dir() -> Path:
    """The cache directory (created on first use). Override with $LIVEEDGE_CACHE_DIR."""
    d = Path(os.environ.get("LIVEEDGE_CACHE_DIR", ".cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def cached_frame(key: str, producer: Callable[[], pd.DataFrame | None]) -> pd.DataFrame | None:
    """Return the cached parquet for ``key``; otherwise call ``producer()``, cache a non-empty
    result, and return it.

    Failures (``None`` or empty) are deliberately NOT cached, so the next run retries them. The
    write goes through a temp file so an interrupted run can't leave a half-written parquet.
    """
    path = cache_dir() / f"{key}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    df = producer()
    if df is not None and len(df) > 0:
        tmp = path.with_name(path.name + ".tmp")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
    return df
