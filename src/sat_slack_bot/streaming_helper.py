"""Streaming helper for Slack bot."""

import asyncio
import logging
import time
from collections.abc import AsyncIterable, AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, BaseMessageChunk, HumanMessage
from slack_bolt.async_app import AsyncSay
from slack_sdk.web.async_client import AsyncWebClient

logger = logging.getLogger(__name__)


@dataclass
class StreamingConfig:
    """Options for streaming messages to Slack."""

    initial_message: BaseMessage = field(
        default_factory=lambda: HumanMessage(content="Thinking...")
    )
    interval: float = 1.0


@dataclass
class StreamState:
    """Represents the state of a streaming operation."""

    ts: str | None = None
    # content가 문자열이면 단순 이어붙임, 블록이면 리스트로 관리
    content: str | list[Any] = ""
    last_update: float = time.time()


DEFAULT_STREAMING_CONFIG = StreamingConfig()


@asynccontextmanager
async def say_streaming(
    say: AsyncSay,
    config: StreamingConfig = DEFAULT_STREAMING_CONFIG,
) -> AsyncIterator[StreamState]:
    """Async context manager for streaming messages to Slack.

    Sends an initial message if provided and yields a StreamState object.
    """
    state = StreamState(ts=None, content="", last_update=time.time())
    content = config.initial_message.content

    if content:
        resp = None
    if isinstance(content, str):
        resp = await say(content)
        state.content = content
    elif isinstance(content, list):
        resp = await say({"blocks": content})
        state.content = content

    if resp and "ts" in resp:
        state.ts = resp["ts"]
        state.last_update = time.time()

    yield state


async def stream_chunks_to_slack(  # noqa: C901, PLR0912, PLR0913, PLR0915
    say: AsyncSay,
    client: AsyncWebClient,
    channel: str,
    thread_ts: str | None,
    chunks: AsyncIterable[BaseMessageChunk],
    config: StreamingConfig = DEFAULT_STREAMING_CONFIG,
) -> None:
    """Stream message chunks to Slack, updating the message in real time.

    Sends a temporary initial message (if provided), concurrently reads chunks
    from the LLM stream, and updates the Slack message until streaming is complete.
    If streaming starts, the initial default message is cleared so that it is not appended to.
    """
    state = StreamState(ts=None, content="", last_update=time.time())
    streaming_done = False

    # Send temporary initial message.
    temp_message = config.initial_message.content
    resp = None
    if temp_message:
        if isinstance(temp_message, str) and temp_message.strip():
            resp = await say(temp_message, thread_ts=thread_ts)
            state.content = temp_message
        elif isinstance(temp_message, list):
            resp = await say({"blocks": temp_message}, thread_ts=thread_ts)
            state.content = temp_message
        if resp and "ts" in resp:
            state.ts = resp["ts"]
            logger.debug("Temporary message sent, ts=%s", state.ts)

    # Helper function for updating (or deleting) the message.
    async def update_message(*, is_final: bool = False) -> None:
        if not state.ts:
            return
        if isinstance(state.content, str):
            text_to_send = state.content.strip()
            if is_final:
                # 최종 업데이트 시 내용이 초기 메시지와 동일하면 삭제
                initial_text = temp_message.strip() if isinstance(temp_message, str) else None
                if initial_text and text_to_send == initial_text:
                    await client.chat_delete(channel=channel, ts=state.ts)
                    return
            if text_to_send:
                await client.chat_update(
                    channel=channel,
                    ts=state.ts,
                    text=text_to_send,
                    blocks=None,
                )
        elif isinstance(state.content, list):
            fallback_text = "\n".join(
                block["text"]["text"]
                for block in state.content
                if block.get("type") == "section" and block.get("text", {}).get("text")
            )
            await client.chat_update(
                channel=channel,
                ts=state.ts,
                text=fallback_text,
                blocks=state.content,
            )

    async def writer() -> None:
        nonlocal streaming_done
        while not streaming_done:
            await asyncio.sleep(config.interval)
            await update_message(is_final=False)
        # 최종 업데이트 시 삭제 조건을 적용합니다.
        await update_message(is_final=True)

    writer_task = asyncio.create_task(writer())

    first_chunk_received = False
    try:
        async for chunk in chunks:
            if not first_chunk_received:
                if isinstance(state.content, str):
                    if (
                        isinstance(temp_message, str)
                        and state.content.strip() == temp_message.strip()
                    ):
                        state.content = ""
                    state.content = ""
                elif isinstance(state.content, list):
                    pass
                first_chunk_received = True
            if isinstance(state.content, str) and isinstance(chunk.content, str):
                state.content += chunk.content
            else:
                logger.warning(
                    "Incompatible chunk or state content types: %s += %s",
                    type(state.content),
                    type(chunk.content),
                )

    except Exception:
        logger.exception("Error during streaming")
    finally:
        streaming_done = True
        await writer_task


async def respond_with_llm_stream(  # noqa: PLR0913
    prompt: str,
    llm: BaseChatModel,
    say: AsyncSay,
    client: AsyncWebClient,
    channel: str,
    thread_ts: str | None = None,
    config: StreamingConfig | None = None,
) -> None:
    """Generate a response using LLM stream and send it to Slack.

    Uses the provided LLM to generate streaming message chunks and forwards them to Slack.
    """
    chunks = llm.astream([HumanMessage(content=prompt)])
    await stream_chunks_to_slack(
        say=say,
        client=client,
        channel=channel,
        thread_ts=thread_ts,
        chunks=chunks,
        config=config or DEFAULT_STREAMING_CONFIG,
    )


async def respond_with_llm_stream_from_event(  # noqa: PLR0913
    event: dict[str, Any],
    prompt: str,
    llm: BaseChatModel,
    say: AsyncSay,
    client: AsyncWebClient,
    thread_ts: str,
    config: StreamingConfig | None = DEFAULT_STREAMING_CONFIG,
) -> None:
    """Generate a response using LLM stream based on an event and send it to Slack."""
    await respond_with_llm_stream(
        prompt=prompt,
        llm=llm,
        say=say,
        client=client,
        channel=event["channel"],
        thread_ts=thread_ts,
        config=config,
    )
