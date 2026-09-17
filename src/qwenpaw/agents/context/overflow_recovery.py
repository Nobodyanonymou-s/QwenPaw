# -*- coding: utf-8 -*-
"""One-shot overflow recovery across model calls and stream consumption.

The agent supplies recovery that classifies the error, compacts context,
rebuilds input, and calls the model again. This module owns the retry boundary
and stream lifecycle; the recovery response is never wrapped for retry again.
"""

from __future__ import annotations

import inspect
import logging
import re
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import aclosing
from typing import Any

from ...providers.stream_progress import has_meaningful_stream_content

logger = logging.getLogger(__name__)

# Providers state their real context limit in overflow rejections, e.g.
# "This model's maximum context length is 8192 tokens." (OpenAI-style),
# "context length is 4096 tokens", or Gemini's "the maximum number of
# tokens allowed (1048576)".
_CONTEXT_LIMIT_RE = re.compile(
    r"maximum context length is (\d+)"
    r"|context length is (\d+) tokens"
    r"|tokens allowed \((\d+)\)",
)


def parse_reported_context_limit(exc: Exception) -> int | None:
    """Return the context limit a provider reported in an overflow 400."""
    match = _CONTEXT_LIMIT_RE.search(str(exc))
    if not match:
        return None
    return int(next(group for group in match.groups() if group))


async def persist_reported_context_limit(
    exc: Exception,
    *,
    provider_id: str,
    model_id: str,
) -> bool:
    """Persist a provider-reported limit so recovery compacts against it.

    The stored window can be a stale, larger belief (e.g. the 128K default
    for uncatalogued models), which keeps compaction from firing at the
    provider's real limit and makes every recovery attempt fail the same
    way. Compaction re-resolves the stored window on its hot path, so
    correcting it *before* recovery lets that same attempt succeed. Best
    effort: parse or persist failures only log.
    """
    limit = parse_reported_context_limit(exc)
    if limit is None or not provider_id or not model_id:
        return False
    try:
        from ...providers.provider_manager import ProviderManager

        await ProviderManager.get_instance().update_model_config(
            provider_id=provider_id,
            model_id=model_id,
            config={"max_input_length": limit},
        )
    except Exception:  # pylint: disable=broad-except
        logger.warning(
            "Could not persist provider-reported context limit %d",
            limit,
            exc_info=True,
        )
        return False
    logger.info(
        "Stored context window for %s/%s corrected to %d tokens from the "
        "provider rejection.",
        provider_id,
        model_id,
        limit,
    )
    return True


async def call_with_overflow_recovery(
    call_model: Callable[..., Awaitable[Any]],
    recover: Callable[[Exception], Awaitable[Any]],
    **kwargs: Any,
) -> Any:
    """Recover errors from invocation or pre-output stream consumption."""
    try:
        response = await call_model(**kwargs)
    except Exception as exc:
        return await recover(exc)
    if inspect.isasyncgen(response):
        return _stream_with_overflow_recovery(response, recover)
    return response


async def _stream_with_overflow_recovery(
    stream: AsyncGenerator[Any, None],
    recover: Callable[[Exception], Awaitable[Any]],
) -> AsyncGenerator[Any, None]:
    """Close consumed streams and never replay meaningful model output."""
    emitted = False
    try:
        async with aclosing(stream):
            async for chunk in stream:
                emitted = emitted or has_meaningful_stream_content(
                    chunk.content,
                )
                yield chunk
        return
    except Exception as exc:
        if emitted:
            raise
        response = await recover(exc)
    # Consume the retry directly so both failure phases share one attempt.
    if inspect.isasyncgen(response):
        async with aclosing(response):
            async for chunk in response:
                yield chunk
    else:
        yield response
