# -*- coding: utf-8 -*-
"""DoomLoopGate: session-safe doom loop detection.

Inherits LoopGate for per-session state isolation.
Includes inline sliding-window similarity detection.
"""
from __future__ import annotations

import hashlib
import json
import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from .base import (
    StopAction,
    StopHandlerResult,
)
from .loop_gate import LoopGate

logger = logging.getLogger(__name__)

_ARGS_TEXT_LIMIT = 256


@dataclass
class _ToolCallRecord:
    """One recorded tool call for pattern analysis."""

    tool_name: str
    args_hash: str
    fingerprint: str = ""


@dataclass
class _DoomState:
    """Per-session doom loop state."""

    history: deque = field(default_factory=deque)
    consecutive_hits: int = 0
    prompt: str = ""
    last_recorded_iter: int = -1


class DoomLoopGate(LoopGate):
    """Multi-stage doom loop gate (session-safe).

    Sliding-window repetition detection that escalates
    through configured stages.

    - action="modify_prompt": INTERRUPT_AND_CONTINUE,
      inject warning via build_continuation().
    - action="stop": return TERMINATE immediately.
    """

    @property
    def name(self) -> str:
        return "doom-loop"

    @property
    def priority(self) -> int:
        return 5

    def __init__(
        self,
        *,
        window_size: int = 3,
        similarity_threshold: float = 1.0,
        fuzzy_streak_warn: int = 5,
        fuzzy_streak_stop: int = 8,
        stages: list | None = None,
    ) -> None:
        super().__init__()
        self._window_size = max(2, window_size)
        self._threshold = similarity_threshold
        self._fuzzy_warn = max(2, fuzzy_streak_warn)
        self._fuzzy_stop = max(self._fuzzy_warn + 1, fuzzy_streak_stop)
        self._stages = sorted(
            stages or [],
            key=lambda s: s.after,
        )

    def _ensure_state(self) -> _DoomState:
        """Get or create per-session state."""
        state = self._state()
        if state is None:
            state = _DoomState(
                history=deque(
                    maxlen=max(
                        self._window_size * 2,
                        self._fuzzy_stop + 1,
                    ),
                ),
            )
            self.activate(state)
        return state

    def record(
        self,
        tool_name: str,
        args_hash: str,
        fingerprint: str = "",
    ) -> None:
        """Record a completed tool call."""
        state = self._ensure_state()
        state.history.append(
            _ToolCallRecord(
                tool_name=tool_name,
                args_hash=args_hash,
                fingerprint=fingerprint,
            ),
        )

    def reset_turn(self) -> None:
        """Clear history and counters for current session."""
        state = self._state()
        if state is not None:
            state.history.clear()
            state.consecutive_hits = 0
            state.prompt = ""
            state.last_recorded_iter = -1

    async def check(
        self,
        ctx: Any,
    ) -> StopHandlerResult:
        """Evaluate doom loop state.

        Auto-records tool calls from agent context when
        available (no explicit record() needed).
        """
        _bypass = StopHandlerResult(
            action=StopAction.BYPASS,
        )
        state = self._ensure_state()
        self._auto_record_from_ctx(ctx, state)

        is_looping = self._detect_repetition(state)

        if not is_looping:
            streak = self._fuzzy_streak(state)
            if streak >= self._fuzzy_stop:
                logger.info(
                    "DoomLoopGate: STOP after fuzzy streak of %d",
                    streak,
                )
                return StopHandlerResult(
                    action=StopAction.TERMINATE,
                    reason=(
                        "Doom loop: agent stuck after "
                        f"{streak} consecutive similar calls"
                    ),
                )
            if streak >= self._fuzzy_warn:
                state.prompt = (
                    "[WARNING] Repetitive pattern detected. You are calling "
                    f"{self._streak_tool(state)} with similar arguments "
                    f"{streak} times without progress. Try a completely "
                    "different approach."
                )
                logger.warning(
                    "DoomLoopGate: warning at fuzzy streak of %d",
                    streak,
                )
                return StopHandlerResult(
                    action=StopAction.INTERRUPT_AND_CONTINUE,
                    reason="doom_loop repetition warning",
                )
            state.consecutive_hits = 0
            state.prompt = ""
            return _bypass

        if state.consecutive_hits == 0:
            state.consecutive_hits = self._window_size
        else:
            state.consecutive_hits += 1

        active_stage = None
        for stage in reversed(self._stages):
            if state.consecutive_hits >= stage.after:
                active_stage = stage
                break

        if active_stage is None:
            return _bypass

        if active_stage.action == "stop":
            logger.info(
                "DoomLoopGate: STOP after %d hits",
                state.consecutive_hits,
            )
            return StopHandlerResult(
                action=StopAction.TERMINATE,
                reason=active_stage.prompt,
            )

        state.prompt = active_stage.prompt
        logger.warning(
            "DoomLoopGate: warning at %d hits",
            state.consecutive_hits,
        )
        return StopHandlerResult(
            action=StopAction.INTERRUPT_AND_CONTINUE,
            reason="doom_loop repetition warning",
        )

    def build_continuation(self) -> str:
        """Return current doom loop warning."""
        state = self._state()
        if state is None:
            return ""
        return state.prompt

    def _auto_record_from_ctx(
        self,
        ctx: Any,
        state: _DoomState,
    ) -> None:
        """Extract latest tool call from agent context."""
        if not isinstance(ctx, dict):
            return
        agent = ctx.get("agent")
        if agent is None:
            return
        cur_iter = ctx.get("iteration", 0)
        if cur_iter <= state.last_recorded_iter:
            return
        state.last_recorded_iter = cur_iter

        context = getattr(
            getattr(agent, "state", None),
            "context",
            [],
        )
        if not context:
            return
        last_msg = context[-1]
        content = getattr(last_msg, "content", None)
        if not content or not isinstance(content, list):
            return
        for block in reversed(content):
            btype = getattr(block, "type", None)
            if isinstance(block, dict):
                btype = block.get("type")
            if btype in ("tool_call", "tool_use"):
                name = (
                    block.get("name", "")
                    if isinstance(block, dict)
                    else getattr(block, "name", "")
                )
                raw_input = (
                    block.get("input", "")
                    if isinstance(block, dict)
                    else getattr(block, "input", "")
                )
                args_hash = self._hash_args(raw_input)
                fingerprint = self._fingerprint_args(raw_input)
                state.history.append(
                    _ToolCallRecord(
                        tool_name=name,
                        args_hash=args_hash,
                        fingerprint=fingerprint,
                    ),
                )
                return

    @staticmethod
    def _hash_args(raw_input: Any) -> str:
        """Hash tool call args with truncation for large inputs.

        Only the first 2048 bytes are hashed — enough
        for repetition detection without serializing
        potentially large file contents.
        """
        _MAX_HASH_INPUT = 2048
        if isinstance(raw_input, str):
            data = raw_input[:_MAX_HASH_INPUT].encode()
        else:
            data = json.dumps(
                raw_input,
                sort_keys=True,
                default=str,
            ).encode()[:_MAX_HASH_INPUT]
        return hashlib.md5(data).hexdigest()[:8]

    @staticmethod
    def _mask_numbers(value: Any) -> Any:
        """Replace numeric values with a placeholder, recursively.

        Only true numbers and digit-only strings are masked; digits inside
        file names or free text are kept so that e.g. ``slide_1.png`` and
        ``slide_2.png`` stay distinct.
        """
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return "#"
        if isinstance(value, str):
            return "#" if value.isdigit() else value
        if isinstance(value, dict):
            return {
                key: DoomLoopGate._mask_numbers(val)
                for key, val in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [DoomLoopGate._mask_numbers(val) for val in value]
        return value

    @classmethod
    def _fingerprint_args(cls, raw_input: Any) -> str:
        """Canonical args text with numbers masked, for streak detection."""
        if isinstance(raw_input, str):
            try:
                raw_input = json.loads(raw_input)
            except (TypeError, ValueError):
                return raw_input[:_ARGS_TEXT_LIMIT]
        return json.dumps(
            cls._mask_numbers(raw_input),
            sort_keys=True,
            default=str,
        )[:_ARGS_TEXT_LIMIT]

    def _detect_repetition(
        self,
        state: _DoomState,
    ) -> bool:
        """Check sliding window for repetition."""
        if len(state.history) < self._window_size:
            return False

        window = list(state.history)[-self._window_size :]
        similarity = self._compute_similarity(window)

        if similarity >= self._threshold:
            logger.warning(
                "Doom loop: sim=%.2f thr=%.2f",
                similarity,
                self._threshold,
            )
            return True
        return False

    @staticmethod
    def _fingerprint(record: _ToolCallRecord) -> str:
        """Stable signature of a call: tool plus masked args.

        Calls that differ only in numeric argument values (offsets, line
        ranges, limits) share a fingerprint. A long run of same-fingerprint
        calls is the natural shape of a loop where a model retries one tool
        with varying numbers — while legitimate work usually interleaves
        other tools or touches different files.
        """
        return f"{record.tool_name}:{record.fingerprint}"

    def _fuzzy_streak(self, state: _DoomState) -> int:
        """Length of the trailing run of same-fingerprint calls."""
        history = list(state.history)
        if not history:
            return 0
        fingerprint = self._fingerprint(history[-1])
        streak = 0
        for record in reversed(history):
            if self._fingerprint(record) != fingerprint:
                break
            streak += 1
        return streak

    def _streak_tool(self, state: _DoomState) -> str:
        """Tool name of the current fuzzy streak (for warning text)."""
        history = list(state.history)
        return history[-1].tool_name if history else "the same tool"

    @staticmethod
    def _compute_similarity(
        window: list[_ToolCallRecord],
    ) -> float:
        """Compute action pattern similarity.

        Formula: 1 - (unique - 1) / (total - 1)

        Precondition: ``len(window) >= 2``.
        Callers must ensure this; ``_detect_repetition``
        guards via ``len(history) < window_size``
        where ``window_size >= 2``.
        """
        if not window or len(window) <= 1:
            return 0.0

        sigs = [f"{r.tool_name}:{r.args_hash}" for r in window]
        unique = len(set(sigs))
        total = len(sigs)
        return 1.0 - (unique - 1) / (total - 1)


__all__ = ["DoomLoopGate"]
