"""Slack bolt app for the SAT Slack Bot."""

import json
import logging
import os
from typing import Any

from dotenv import load_dotenv
from langchain.chains import ConversationChain
from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp, AsyncSay
from slack_sdk.web.async_client import AsyncWebClient

from sat_slack_bot.memory import slack_full_thread_to_memory
from sat_slack_bot.provider import llm

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SLACK_BOT_TOKEN = os.environ.get("SLACK_BOT_TOKEN", "")

app = AsyncApp(token=SLACK_BOT_TOKEN)


@app.event("app_mention")
async def handle_app_mention(
    event: dict[str, Any],
    context: dict[str, Any],
    client: AsyncWebClient,
    say: AsyncSay,
) -> None:
    """Handle app mention events."""
    logger.debug("Received context: %s", context)
    logger.debug("Received event: %s", json.dumps(event, indent=2))

    thread_ts = event.get("thread_ts") or event["ts"]

    prompt = event.get("text", "")
    if not prompt:
        return

    bot_user_id = context["bot_user_id"]

    memory = await slack_full_thread_to_memory(
        client=client,
        channel=event["channel"],
        thread_ts=thread_ts,
        bot_user_id=bot_user_id,
        llm=llm,
    )

    chain = ConversationChain(llm=llm, memory=memory)

    response = await chain.arun(prompt)
    await say(text=response, thread_ts=thread_ts)


@app.event("message")
async def handle_dm(event: dict[str, Any], say: AsyncSay) -> None:
    """Handle direct messages."""
    if event.get("channel_type") == "im":
        logger.info("Received DM: %s", event)
        await say("I received your DM! :wave:")


async def main() -> None:
    """Start the Slack app."""
    handler = AsyncSocketModeHandler(app, os.environ["SLACK_APP_TOKEN"])
    await handler.start_async()  # type: ignore[no-untyped-call]


@app.event({"type": "event_callback"})
async def catch_all_events(body: dict[str, Any]) -> None:
    """Catch all events."""
    logger.info("Got an event callback:")
    logger.info(body)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
