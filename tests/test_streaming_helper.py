"""Test SAT Streaming Helper."""

import asyncio
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

import pytest
from langchain_core.messages import BaseMessageChunk, HumanMessage

if TYPE_CHECKING:
    from slack_bolt.context.say.async_say import AsyncSay as SlackAsyncSay
    from slack_sdk.web.async_client import AsyncWebClient as SlackAsyncWebClient
from sat_slack_bot.streaming_helper import (
    StreamingConfig,
    say_streaming,
    stream_chunks_to_slack,
)


class DummySay:
    """Dummy say class for testing."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str | None]] = []

    async def __call__(self, content: str, thread_ts: str | None = None) -> dict[str, str]:
        """Simulate sending a message to Slack."""
        self.calls.append((content, thread_ts))
        return {"ts": "12345"}


class DummyClient:
    """Dummy client class for testing."""

    def __init__(self) -> None:
        self.update_calls: list[dict[str, Any]] = []
        self.delete_calls: list[tuple[str, str]] = []

    async def chat_update(
        self, channel: str, ts: str, text: str, blocks: dict[str, Any]
    ) -> dict[str, Any]:
        """Simulate updating a message in Slack."""
        self.update_calls.append(
            {
                "channel": channel,
                "ts": ts,
                "text": text,
                "blocks": blocks,
            }
        )
        return {}

    async def chat_delete(self, channel: str, ts: str) -> dict[str, Any]:
        """Simulate deleting a message in Slack."""
        self.delete_calls.append((channel, ts))
        return {}


class DummyChunk(BaseMessageChunk):
    """Dummy chunk class for testing."""

    def __init__(self, content: str) -> None:
        self.content = content


async def generate_chunks() -> AsyncIterator[DummyChunk]:
    """Generate dummy chunks for testing."""
    yield DummyChunk("Hello, ")
    await asyncio.sleep(0.5)
    yield DummyChunk("world!")


class DummyLLM:
    """Dummy LLM class for testing."""

    def astream(self, messages: Any) -> AsyncIterator[DummyChunk]:
        """Simulate streaming messages from an LLM."""
        return generate_chunks()


@pytest.fixture
def dummy_say() -> DummySay:
    """Fixture for DummySay."""
    return DummySay()


@pytest.fixture
def dummy_client() -> DummyClient:
    """Fixture for DummyClient."""
    return DummyClient()


@pytest.fixture
def dummy_llm() -> DummyLLM:
    """Fixture for DummyLLM."""
    return DummyLLM()


@pytest.mark.asyncio
async def test_say_streaming(dummy_say: DummySay) -> None:
    """Test say_streaming function."""
    config = StreamingConfig(initial_message=HumanMessage(content="Test Initial"))
    async with say_streaming(cast("SlackAsyncSay", dummy_say), config) as state:
        assert state.ts == "12345"
        assert state.content == "Test Initial"
        assert isinstance(state.last_update, float)

    assert len(dummy_say.calls) == 1


@pytest.mark.asyncio
async def test_stream_chunks_to_slack_no_chunks(
    dummy_say: DummySay, dummy_client: DummyClient
) -> None:
    """Test stream_chunks_to_slack with no chunks.

    - test if the message is deleted after streaming
    - test if the delete message is called
    """

    async def empty_chunks() -> AsyncIterator[DummyChunk]:
        empty: list[DummyChunk] = []
        for item in empty:
            yield item

    config = StreamingConfig(initial_message=HumanMessage(content="Thinking..."), interval=0.01)

    await stream_chunks_to_slack(
        say=cast("SlackAsyncSay", dummy_say),
        client=cast("SlackAsyncWebClient", dummy_client),
        channel="C123",
        thread_ts=None,
        chunks=empty_chunks(),
        config=config,
    )
    # Check if chat_delete was called.
    assert len(dummy_client.delete_calls) >= 1


@pytest.mark.asyncio
async def test_stream_chunks_to_slack(
    dummy_say: DummySay, dummy_client: DummyClient, dummy_llm: DummyLLM
) -> None:
    """Test stream_chunks_to_slack function.

    - test if the initial message is sent
    - test if the message is updated with chunks
    - test if the message is deleted after streaming
    """
    config = StreamingConfig(initial_message=HumanMessage(content="Thinking..."), interval=0.01)
    await stream_chunks_to_slack(
        say=cast("SlackAsyncSay", dummy_say),
        client=cast("SlackAsyncWebClient", dummy_client),
        channel="C123",
        thread_ts=None,
        chunks=generate_chunks(),
        config=config,
    )

    assert len(dummy_client.delete_calls) == 0

    updated_texts = [call["text"] for call in dummy_client.update_calls if call["text"]]
    assert any("Hello, world!" in text for text in updated_texts)
