"""Provides an interface to interact with the ChatLiteLLM model."""

import os

from langchain_community.chat_models import ChatLiteLLM

model = os.environ.get("LLM_MODEL", "openai/gpt-4o")
temperature = float(os.environ.get("LLM_TEMPERATURE", "0"))
api_base = os.environ.get("LLM_API_BASE", "https://api.openai.com/v1")

llm = ChatLiteLLM(model=model, temperature=temperature, api_base=api_base, streaming=True)
