# -*- coding: utf-8 -*-
"""Tests for SSE delta coalescing."""

from __future__ import annotations

import json
from typing import Any

import pytest

from qwenpaw.app.sse_coalescer import SSECoalescer
from qwenpaw.app.task_tracker import TaskTracker


def _sse(evt: dict[str, Any]) -> str:
    return f"data: {json.dumps(evt)}\n\n"


def _text(msg_id: str, text: str, index: int = 0) -> dict[str, Any]:
    return {
        "object": "content",
        "type": "text",
        "delta": True,
        "msg_id": msg_id,
        "index": index,
        "text": text,
        "sequence_number": 7,
    }


def _barrier(msg_id: str) -> dict[str, Any]:
    return {
        "object": "message",
        "type": "message",
        "msg_id": msg_id,
        "status": "completed",
        "sequence_number": 8,
    }


def _run(frames: list[str], **kwargs: Any) -> list[dict[str, Any]]:
    co = SSECoalescer(**kwargs)
    out: list[str] = []
    for f in frames:
        out.extend(co.push(f))
    out.extend(co.flush())
    return [json.loads(f[5:].strip()) for f in out]


def test_consecutive_text_deltas_merge_until_barrier():
    out = _run(
        [
            _sse(_text("m1", "a")),
            _sse(_text("m1", "b")),
            _sse(_text("m1", "c")),
            _sse(_barrier("m1")),
        ],
    )

    assert len(out) == 2
    assert out[0]["text"] == "abc"
    assert out[0]["delta"] is True
    # the merged event keeps the first delta's sequence number
    assert out[0]["sequence_number"] == 7
    assert out[1]["status"] == "completed"


def test_different_block_flushes_previous():
    """Only same-block deltas merge; a different block never concatenates."""
    out = _run(
        [
            _sse(_text("m1", "a")),
            _sse(_text("m1", "b", index=1)),
            _sse(_text("m2", "c")),
        ],
    )

    # three distinct keys -> three separate events, order preserved
    assert [e["text"] for e in out] == ["a", "b", "c"]
    assert out[0]["index"] == 0
    assert out[1]["index"] == 1
    assert out[2]["msg_id"] == "m2"


def test_max_bytes_bounds_each_merged_event():
    co = SSECoalescer(max_bytes=10)
    out: list[str] = []
    for i in range(5):
        out.extend(co.push(_sse(_text("m1", "x" * 4))))
    out.extend(co.flush())

    payloads = [json.loads(f[5:].strip()) for f in out]
    assert "".join(e["text"] for e in payloads) == "x" * 20
    assert all(len(e["text"]) <= 14 for e in payloads)


def test_tool_call_argument_fragments_merge():
    frames = [
        _sse(
            {
                "object": "content",
                "type": "data",
                "delta": True,
                "msg_id": "m1",
                "index": 0,
                "data": {"arguments": '{"path"'},
            },
        ),
        _sse(
            {
                "object": "content",
                "type": "data",
                "delta": True,
                "msg_id": "m1",
                "index": 0,
                "data": {"arguments": ': "big.md"}'},
            },
        ),
    ]
    out = _run(frames)

    assert len(out) == 1
    assert out[0]["data"]["arguments"] == '{"path": "big.md"}'


def test_non_delta_snapshots_are_barriers():
    snapshot = {
        "object": "content",
        "type": "data",
        "delta": False,
        "msg_id": "m1",
        "index": 0,
        "data": {"output": "full result"},
    }
    out = _run(
        [_sse(_text("m1", "a")), _sse(snapshot), _sse(_text("m1", "b"))]
    )

    assert len(out) == 3
    assert out[1] == snapshot


def test_heartbeat_passthrough_by_default_dropped_on_demand():
    hb = {"object": "message", "type": "heartbeat"}

    assert len(_run([_sse(hb)])) == 1
    assert _run([_sse(hb), _sse(hb)], drop_heartbeats=True) == []


def test_unparseable_frame_passes_through_and_flushes():
    out_frames: list[str] = []
    co = SSECoalescer()
    out_frames.extend(co.push(_sse(_text("m1", "a"))))
    out_frames.extend(co.push("data: not-json\n\n"))

    assert len(out_frames) == 2
    assert out_frames[1] == "data: not-json\n\n"


def test_equivalence_concat_text_matches_input():
    parts = [f"chunk{i}-" for i in range(50)]
    frames = [_sse(_text("m1", p)) for p in parts]
    out = _run(frames + [_sse(_barrier("m1"))])

    merged = "".join(
        e["text"]
        for e in out
        if e.get("object") == "content" and e.get("delta")
    )
    assert merged == "".join(parts)
    # barriers keep their relative position
    assert out[-1]["status"] == "completed"


@pytest.mark.asyncio
async def test_producer_buffers_coalesced_events():
    """The tracker's buffer and subscribers see merged deltas."""
    parts = [f"t{i};" for i in range(20)]
    frames = [_sse(_text("m1", p)) for p in parts]

    async def stream_fn(_payload: Any):
        for f in frames:
            yield f

    tracker = TaskTracker()
    queue, is_new = await tracker.attach_or_start("k", None, stream_fn)
    assert is_new
    received = []
    while True:
        item = await queue.get()
        if item is None:
            break
        received.append(item)

    assert len(received) < len(frames)
    merged = "".join(
        json.loads(f[5:].strip())["text"]
        for f in received
        if '"delta": true' in f
    )
    assert merged == "".join(parts)
