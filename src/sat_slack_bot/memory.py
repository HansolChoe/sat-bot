"""Module for memory management in the SAT Slack Bot."""

from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import AIMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from slack_sdk.web.async_client import AsyncWebClient


async def slack_full_thread_to_memory(
    client: AsyncWebClient,
    channel: str,
    thread_ts: str,
    bot_user_id: str,
    llm: BaseChatModel,
) -> ConversationSummaryBufferMemory:
    """Fetch a Slack thread and stores its messages in a memory buffer."""
    memory = ConversationSummaryBufferMemory(
        llm=llm,
        return_messages=True,
        max_token_limit=3000,
    )
    res = await client.conversations_replies(channel=channel, ts=thread_ts)
    messages = res["messages"]

    for msg in messages:
        user = msg.get("user")
        text = msg.get("text", "")

        if not user or not text:
            continue

        if user == bot_user_id:
            memory.chat_memory.add_message(AIMessage(content=text))
        else:
            memory.chat_memory.add_message(HumanMessage(content=text))

    return memory
