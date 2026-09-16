# -*- coding: utf-8 -*-
"""Coalesce adjacent SSE delta events for streaming responses.

The console frontend appends string payloads from delta events, so a run
of consecutive deltas for the same block is semantically equivalent to a
single delta carrying their concatenation.  Merging such runs before they
reach subscribers and the reconnect replay buffer keeps long token
streams from flooding clients with thousands of tiny events each of
which the UI re-renders for.

Any event that is not a mergeable delta (block ends, completed messages,
tool results, errors, heartbeats when kept) acts as a barrier: pending
merges are flushed first and the barrier passes through unchanged, so
ordering and non-delta semantics are preserved exactly.
"""

from __future__ import annotations

import json
import time
from typing import Any

# Emit a merged event once it reaches this size so one merge cannot grow
# without bound either.
DEFAULT_MAX_BYTES = 8192

# Emit a pending merge once it is this old, so an actively streaming reply
# still renders incrementally instead of arriving as one block.
DEFAULT_WINDOW_SECONDS = 0.1


def _frame(payload: str) -> str:
    return f"data: {payload}\n\n"


def _payload_of(sse: str) -> Any | None:
    """Parse the ``data:`` JSON of one SSE frame; None if unparseable."""
    for line in sse.split("\n"):
        if line.startswith("data:"):
            try:
                return json.loads(line[5:].strip())
            except json.JSONDecodeError:
                return None
    return None


def _is_heartbeat(evt: dict[str, Any]) -> bool:
    return evt.get("object") == "message" and evt.get("type") == "heartbeat"


def _merge_field(evt: dict[str, Any]) -> tuple[str, str] | None:
    """Return ``(field, value)`` when *evt* is a mergeable delta.

    Mergeable deltas are ``content`` events with ``delta=true`` whose
    payload is a plain string the client appends:

    - ``text`` / ``thinking`` blocks append ``text``;
    - tool-call argument fragments append ``data.arguments``.
    """
    if evt.get("object") != "content" or evt.get("delta") is not True:
        return None
    evt_type = evt.get("type")
    if evt_type in ("text", "thinking"):
        value = evt.get("text")
        if isinstance(value, str):
            return "text", value
        return None
    if evt_type == "data":
        data = evt.get("data")
        if isinstance(data, dict):
            args = data.get("arguments")
            if isinstance(args, str):
                return "arguments", args
    return None


def _merge_key(evt: dict[str, Any]) -> tuple:
    return (evt.get("msg_id") or "", evt.get("type"), evt.get("index", 0))


class SSECoalescer:
    """Merge consecutive same-block delta SSE frames.

    One pending merge at a time: a delta for a different block flushes the
    current one first, which keeps emit order identical to input order by
    construction.  ``max_bytes`` bounds each merged event; a barrier (or
    :meth:`flush`) always releases what is pending.
    """

    def __init__(
        self,
        max_bytes: int = DEFAULT_MAX_BYTES,
        *,
        window_s: float = DEFAULT_WINDOW_SECONDS,
        drop_heartbeats: bool = False,
    ) -> None:
        self._max_bytes = max_bytes
        self._window = window_s
        self._drop_heartbeats = drop_heartbeats
        self._key: tuple | None = None
        self._evt: dict[str, Any] | None = None
        self._size = 0
        self._started = 0.0

    def push(self, sse: str) -> list[str]:
        """Feed one SSE frame; return the frames that may be emitted now."""
        evt = _payload_of(sse)
        if not isinstance(evt, dict):
            # unparseable or non-object payloads (e.g. plain numbers)
            # are passed through untouched
            return self.flush() + [sse]

        if self._drop_heartbeats and _is_heartbeat(evt):
            return []

        merge = _merge_field(evt)
        if merge is None:
            return self.flush() + [sse]

        out: list[str] = []
        if self._evt is not None and (
            _merge_key(evt) != self._key
            or time.monotonic() - self._started >= self._window
        ):
            out = self.flush()

        field, value = merge
        if self._evt is None:
            self._key = _merge_key(evt)
            self._evt = evt
            self._size = len(sse)
            self._started = time.monotonic()
            return out

        if field == "arguments":
            data = self._evt.setdefault("data", {})
            data["arguments"] = data.get("arguments", "") + value
        else:
            self._evt[field] = self._evt.get(field, "") + value
        self._size += len(value)
        if self._size >= self._max_bytes:
            out.extend(self.flush())
        return out

    def flush(self) -> list[str]:
        """Release the pending merge, if any."""
        if self._evt is None:
            return []
        payload = json.dumps(self._evt, ensure_ascii=False)
        self._evt = None
        self._key = None
        self._size = 0
        self._started = 0.0
        return [_frame(payload)]
