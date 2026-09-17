# -*- coding: utf-8 -*-
# pylint: disable=protected-access,unused-argument
"""Tests for the one-shot Scroll recovery on provider context overflow."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from agentscope.agent import Agent
from google.genai import errors as genai_errors

from qwenpaw.agents.context.overflow_recovery import (
    parse_reported_context_limit,
)
from qwenpaw.agents.react_agent import QwenPawAgent


class _ContextOverflowError(Exception):
    status_code = 400


class _ScrollManager:
    def __init__(self, refreshed_context):
        self.refreshed_context = refreshed_context
        self.calls = []

    async def recover_from_context_overflow(self, agent):
        self.calls.append(agent)
        agent.state.context = list(self.refreshed_context)
        return True

    async def compress(self, agent, context_config):
        raise AssertionError("normal compression must not be used")

    def on_save(self, agent, blocks):
        raise AssertionError("on_save must not be used")


class _OtherContextManager:
    def __init__(self):
        self.calls = []

    async def compress(self, agent, context_config):
        self.calls.append((agent, context_config))

    def on_save(self, agent, blocks):
        self.calls.append((agent, blocks))


class _NoOpScrollManager(_ScrollManager):
    async def recover_from_context_overflow(self, agent):
        self.calls.append(agent)
        return False


def _agent(*, context_manager=None):
    agent = object.__new__(QwenPawAgent)
    agent._context_manager = context_manager
    agent.state = SimpleNamespace(context=["old-1", "old-2"])
    agent._prepare_model_input = AsyncMock(
        return_value={
            "messages": ["system", "compacted"],
            "tools": [{"name": "tool-after-compact"}],
        },
    )
    return agent


@pytest.mark.asyncio
async def test_context_overflow_forces_scroll_rebuilds_and_retries_once(
    monkeypatch,
):
    calls = []

    async def fake_call_model(self, messages, tools, tool_choice=None):
        calls.append((messages, tools, tool_choice))
        if len(calls) == 1:
            raise _ContextOverflowError(
                "Error code: 400 - Range of input length should be "
                "[1, 983616]",
            )
        return "ok"

    monkeypatch.setattr(Agent, "_call_model", fake_call_model)
    scroll = _ScrollManager(["compacted"])
    agent = _agent(context_manager=scroll)
    agent.compress_context = AsyncMock(wraps=agent.compress_context)

    result = await agent._call_model(
        messages=["system", "old-1", "old-2"],
        tools=[{"name": "old-tool"}],
        tool_choice="auto",
    )

    assert result == "ok"
    assert len(scroll.calls) == 1
    assert scroll.calls[0] is agent
    agent.compress_context.assert_not_awaited()
    agent._prepare_model_input.assert_awaited_once()
    assert calls == [
        (
            ["system", "old-1", "old-2"],
            [{"name": "old-tool"}],
            "auto",
        ),
        (
            ["system", "compacted"],
            [{"name": "tool-after-compact"}],
            "auto",
        ),
    ]


@pytest.mark.asyncio
async def test_context_overflow_retries_only_once(monkeypatch):
    calls = 0

    async def fake_call_model(self, messages, tools, tool_choice=None):
        nonlocal calls
        calls += 1
        raise _ContextOverflowError(
            "Error code: 400 - context length exceeded",
        )

    monkeypatch.setattr(Agent, "_call_model", fake_call_model)
    scroll = _ScrollManager(["compacted"])
    agent = _agent(context_manager=scroll)

    with pytest.raises(_ContextOverflowError, match="context length"):
        await agent._call_model(messages=["old"], tools=[])

    assert calls == 2
    assert len(scroll.calls) == 1
    agent._prepare_model_input.assert_awaited_once()


@pytest.mark.asyncio
async def test_context_overflow_skips_retry_when_input_is_unchanged(
    monkeypatch,
):
    calls = 0

    async def fake_call_model(self, messages, tools, tool_choice=None):
        nonlocal calls
        calls += 1
        raise _ContextOverflowError(
            "Error code: 400 - context length exceeded",
        )

    monkeypatch.setattr(Agent, "_call_model", fake_call_model)
    manager = _NoOpScrollManager(["unused"])
    agent = _agent(context_manager=manager)

    with pytest.raises(_ContextOverflowError, match="context length"):
        await agent._call_model(
            messages=["system", "old"],
            tools=[{"name": "same-tool"}],
        )

    assert calls == 1
    assert len(manager.calls) == 1
    agent._prepare_model_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_unrelated_bad_request_does_not_compact_or_retry(monkeypatch):
    calls = 0

    async def fake_call_model(self, messages, tools, tool_choice=None):
        nonlocal calls
        calls += 1
        raise _ContextOverflowError(
            "Error code: 400 - unsupported parameter: temperature",
        )

    monkeypatch.setattr(Agent, "_call_model", fake_call_model)
    scroll = _ScrollManager(["compacted"])
    agent = _agent(context_manager=scroll)

    with pytest.raises(_ContextOverflowError, match="unsupported parameter"):
        await agent._call_model(messages=["old"], tools=[])

    assert calls == 1
    assert not scroll.calls
    agent._prepare_model_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_context_overflow_without_scroll_does_not_retry(monkeypatch):
    calls = 0

    async def fake_call_model(self, messages, tools, tool_choice=None):
        nonlocal calls
        calls += 1
        raise _ContextOverflowError(
            "Error code: 400 - prompt is too long",
        )

    monkeypatch.setattr(Agent, "_call_model", fake_call_model)
    agent = _agent(context_manager=None)

    with pytest.raises(_ContextOverflowError, match="prompt is too long"):
        await agent._call_model(messages=["old"], tools=[])

    assert calls == 1
    agent._prepare_model_input.assert_not_awaited()


@pytest.mark.asyncio
async def test_context_overflow_with_other_manager_does_not_retry(monkeypatch):
    calls = 0

    async def fake_call_model(self, messages, tools, tool_choice=None):
        nonlocal calls
        calls += 1
        raise _ContextOverflowError(
            "Error code: 400 - prompt is too long",
        )

    monkeypatch.setattr(Agent, "_call_model", fake_call_model)
    manager = _OtherContextManager()
    agent = _agent(context_manager=manager)

    with pytest.raises(_ContextOverflowError, match="prompt is too long"):
        await agent._call_model(messages=["old"], tools=[])

    assert calls == 1
    assert not manager.calls
    agent._prepare_model_input.assert_not_awaited()


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Error code: 400 - maximum context length is 128000", True),
        ("<400> input length should be between 1 and 983616", False),
        ("Error code: 400 - image_url is unsupported", False),
        ("Error code: 500 - context length exceeded", False),
    ],
)
def test_context_overflow_classifier_requires_400_and_specific_marker(
    message,
    expected,
):
    exc = Exception(message)
    assert QwenPawAgent._is_context_overflow_error(exc) is expected


@pytest.mark.parametrize(
    "message",
    [
        (
            "The input token count (1337419) exceeds the maximum number "
            "of tokens allowed (1048576)."
        ),
        (
            "Unable to submit request because the input token count is "
            "135538 but model only supports up to 32768"
        ),
    ],
)
def test_context_overflow_classifier_supports_gemini_client_error(message):
    exc = genai_errors.ClientError(
        400,
        {
            "error": {
                "status": "INVALID_ARGUMENT",
                "message": message,
            },
        },
    )
    assert QwenPawAgent._is_context_overflow_error(exc) is True


def test_context_overflow_classifier_supports_aiohttp_response_status():
    exc = Exception("context length exceeded")
    exc.response = SimpleNamespace(status=400)
    assert QwenPawAgent._is_context_overflow_error(exc) is True


# ----- provider-reported context limit persistence -----

@pytest.mark.parametrize(
    "message,expected",
    [
        (
            "Error code: 400 - This model's maximum context length is "
            "8192 tokens.",
            8192,
        ),
        ("Error code: 400 - context length is 4096 tokens", 4096),
        (
            "The input token count (1337419) exceeds the maximum number "
            "of tokens allowed (1048576).",
            1048576,
        ),
        ("Error code: 400 - context length exceeded", None),
    ],
)
def test_parse_reported_context_limit_variants(message, expected):
    assert parse_reported_context_limit(Exception(message)) == expected


def _agent_with_model_slot(context_manager, order):
    agent = _agent(context_manager=context_manager)
    agent._agent_config = SimpleNamespace(
        active_model=SimpleNamespace(
            provider_id="prov-a",
            model="model-b",
        ),
    )
    agent._order = order
    return agent


@pytest.mark.asyncio
async def test_reported_context_limit_persisted_before_compaction(
    monkeypatch,
):
    order = []
    update_model_config = AsyncMock(return_value=SimpleNamespace())

    class _FakeProviderManager:
        @staticmethod
        def get_instance():
            return SimpleNamespace(update_model_config=update_model_config)

    monkeypatch.setattr(
        "qwenpaw.providers.provider_manager.ProviderManager",
        _FakeProviderManager,
    )

    async def fake_call_model(self, messages, tools, tool_choice=None):
        raise _ContextOverflowError(
            "This model's maximum context length is 8192 tokens.",
        )

    monkeypatch.setattr(Agent, "_call_model", fake_call_model)

    class _OrderedScroll(_ScrollManager):
        async def recover_from_context_overflow(self, agent):
            order.append("compact")
            return await super().recover_from_context_overflow(agent)

    agent = _agent_with_model_slot(_OrderedScroll(["compacted"]), order)
    update_model_config.side_effect = lambda **kw: order.append("persist")

    async def run():
        try:
            await agent._call_model(
                messages=["system", "old-1"],
                tools=[{"name": "old-tool"}],
                tool_choice="auto",
            )
        except _ContextOverflowError:
            # The retry through the fake raises the same class; only the
            # ordering of persist-vs-compact matters here.
            pass

    await run()

    assert order == ["persist", "compact"]
    update_model_config.assert_awaited_once_with(
        provider_id="prov-a",
        model_id="model-b",
        config={"max_input_length": 8192},
    )


@pytest.mark.asyncio
async def test_reported_context_limit_absent_skips_persist(monkeypatch):
    update_model_config = AsyncMock()

    class _FakeProviderManager:
        @staticmethod
        def get_instance():
            return SimpleNamespace(update_model_config=update_model_config)

    monkeypatch.setattr(
        "qwenpaw.providers.provider_manager.ProviderManager",
        _FakeProviderManager,
    )

    async def fake_call_model(self, messages, tools, tool_choice=None):
        raise _ContextOverflowError(
            "Error code: 400 - context length exceeded",
        )

    monkeypatch.setattr(Agent, "_call_model", fake_call_model)

    agent = _agent_with_model_slot(_ScrollManager(["compacted"]), [])
    with pytest.raises(_ContextOverflowError):
        await agent._call_model(
            messages=["system", "old-1"],
            tools=[{"name": "old-tool"}],
            tool_choice="auto",
        )

    update_model_config.assert_not_awaited()


@pytest.mark.asyncio
async def test_persist_failure_does_not_break_recovery(monkeypatch):
    class _ExplodingProviderManager:
        @staticmethod
        def get_instance():
            raise RuntimeError("provider store unavailable")

    monkeypatch.setattr(
        "qwenpaw.providers.provider_manager.ProviderManager",
        _ExplodingProviderManager,
    )

    calls = []

    async def fake_call_model(self, messages, tools, tool_choice=None):
        calls.append(messages)
        if len(calls) == 1:
            raise _ContextOverflowError(
                "This model's maximum context length is 8192 tokens.",
            )
        return "ok"

    monkeypatch.setattr(Agent, "_call_model", fake_call_model)

    agent = _agent_with_model_slot(_ScrollManager(["compacted"]), [])
    result = await agent._call_model(
        messages=["system", "old-1"],
        tools=[{"name": "old-tool"}],
        tool_choice="auto",
    )

    assert result == "ok"
    assert len(calls) == 2
