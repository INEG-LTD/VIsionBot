"""
General in-memory store for cross-step automation memory.

Features:
- Namespaced items (e.g., kind='clicked', kind='filled_field').
- Scopes: 'global', 'domain', 'page' based on current URL.
- TTL expiration and LRU-like pruning to avoid unbounded growth.
- Simple add/has/get/remove APIs designed for general usage across tasks.
"""
from __future__ import annotations

import time
import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse


def _now() -> float:
    return time.time()


def _domain_of(url: str) -> str:
    try:
        p = urlparse(url or "")
        return (p.netloc or "").lower()
    except Exception:
        return ""


def stable_sig(text: str) -> str:
    """Stable hash for arbitrary text."""
    try:
        return hashlib.sha256((text or "").encode("utf-8")).hexdigest()
    except Exception:
        return text or ""


@dataclass
class MemoryItem:
    kind: str
    signature: str
    created_at: float
    expires_at: Optional[float]
    data: Dict[str, Any]


class MemoryStore:
    """Simple in-memory store with scope isolation and TTL.

    Keys are (scope_key, kind, signature) and values are MemoryItem.
    Scopes:
      - global: shared across all pages/domains
      - domain: per hostname (example.com)
      - page: per exact URL (including path/query)
    """

    def __init__(self, max_items: int = 5000):
        self._max = int(max_items)
        self._store: Dict[Tuple[str, str, str], MemoryItem] = {}

    # ----------------- scope helpers -----------------
    def _scope_key(self, scope: str, url: Optional[str]) -> str:
        s = (scope or "global").lower()
        if s == "global":
            return "global"
        if s == "domain":
            return f"domain::{_domain_of(url or '')}"
        if s == "page":
            return f"page::{(url or '').strip()}"
        # fallback
        return "global"

    # ----------------- maintenance -------------------
    def _purge_expired(self) -> None:
        now = _now()
        for k in list(self._store.keys()):
            item = self._store.get(k)
            if not item:
                continue
            if item.expires_at and now >= item.expires_at:
                try:
                    del self._store[k]
                except Exception:
                    pass

    def _prune_if_needed(self) -> None:
        if len(self._store) <= self._max:
            return
        # Simple prune: drop oldest 10% by created_at
        items = sorted(self._store.items(), key=lambda kv: kv[1].created_at)
        to_drop = max(1, int(len(items) * 0.1))
        for i in range(to_drop):
            try:
                del self._store[items[i][0]]
            except Exception:
                pass

    # ----------------- public API --------------------
    def put(self, kind: str, signature: str, *, scope: str = "domain", url: Optional[str] = None,
            ttl_seconds: Optional[float] = 3600.0, data: Optional[Dict[str, Any]] = None) -> None:
        self._purge_expired()
        skey = self._scope_key(scope, url)
        expires = (_now() + float(ttl_seconds)) if ttl_seconds else None
        key = (skey, kind, signature)
        self._store[key] = MemoryItem(kind=kind, signature=signature, created_at=_now(), expires_at=expires, data=data or {})
        self._prune_if_needed()

    def has(self, kind: str, signature: str, *, scope: str = "domain", url: Optional[str] = None) -> bool:
        self._purge_expired()
        skey = self._scope_key(scope, url)
        return (skey, kind, signature) in self._store

    def get(self, kind: str, signature: str, *, scope: str = "domain", url: Optional[str] = None) -> Optional[Dict[str, Any]]:
        self._purge_expired()
        skey = self._scope_key(scope, url)
        item = self._store.get((skey, kind, signature))
        return item.data if item else None

    def remove(self, kind: str, signature: str, *, scope: str = "domain", url: Optional[str] = None) -> None:
        skey = self._scope_key(scope, url)
        try:
            del self._store[(skey, kind, signature)]
        except Exception:
            pass

    def list(self, kind: Optional[str] = None, *, scope: str = "domain", url: Optional[str] = None) -> Dict[str, MemoryItem]:
        self._purge_expired()
        skey = self._scope_key(scope, url)
        out: Dict[str, MemoryItem] = {}
        for (sc, k, sig), item in self._store.items():
            if sc != skey:
                continue
            if kind and k != kind:
                continue
            out[sig] = item
        return out

    def clear_scope(self, scope: str = "domain", url: Optional[str] = None) -> None:
        skey = self._scope_key(scope, url)
        for key in [k for k in self._store.keys() if k[0] == skey]:
            try:
                del self._store[key]
            except Exception:
                pass

