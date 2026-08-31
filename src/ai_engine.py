"""
Cerebras LLM client for GDELT platform.

Talks to the OpenAI-compatible chat/completions endpoint with plain requests
instead of going through llama-index. That matters for cost control: the
llama-index wrapper sent no output limit, so every call was free to run to
the model's 40,960-token ceiling. gpt-oss-120b is a reasoning model and bills
its hidden chain-of-thought as completion tokens at $0.75/M, so uncapped
calls quietly burned roughly 50x what the old non-reasoning llama3.1-8b did.
Owning the request body lets us cap output and ask for minimal reasoning.

Dropping llama-index also takes transformers/tokenizers/nltk off the import
path, about 40MB of RSS on the Streamlit container (measured: 129MB -> 89MB).
"""

import os
import logging

import requests
import streamlit as st

from src.config import CEREBRAS_MODEL

logger = logging.getLogger("gdelt")

API_URL = "https://api.cerebras.ai/v1/chat/completions"

# The UI renders a 3-5 sentence answer plus a short bullet list. 1200 tokens
# is generous for that while still bounding a reasoning model that would
# otherwise be entitled to 40,960. Kept above ~800 on purpose: reasoning
# tokens are drawn from the same budget, and too tight a cap gets spent on
# thinking and returns an empty answer.
MAX_COMPLETION_TOKENS = 1200

# Cheapest useful setting for "summarise these rows". Sent only when the
# endpoint accepts it - see _post().
REASONING_EFFORT = "low"

AI_AVAILABLE = True


class LLMUnavailable(RuntimeError):
    """Cerebras refused the request (out of credit, rate limited, down)."""


class CerebrasLLM:
    """Minimal stand-in for the llama-index client.

    Only .complete(prompt) -> str is used by the app (ai_chat + rag_engine),
    so that is all this implements.
    """

    def __init__(self, api_key, model=CEREBRAS_MODEL, temperature=0.1):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        # Flipped off permanently if the endpoint rejects reasoning_effort,
        # so we retry once and not on every later call.
        self._send_reasoning_effort = True

    def _post(self, body, timeout):
        return requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=timeout,
        )

    def complete(self, prompt, timeout=60):
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
        }
        if self._send_reasoning_effort:
            body["reasoning_effort"] = REASONING_EFFORT

        try:
            r = self._post(body, timeout)
        except requests.exceptions.Timeout:
            raise LLMUnavailable("AI request timed out. Try again.") from None
        except requests.exceptions.RequestException as e:
            raise LLMUnavailable(f"Could not reach the AI service: {e}") from None

        # reasoning_effort is not advertised in the model's supported_parameters,
        # so treat a 400 that names it as "this endpoint does not take it" and
        # fall back permanently rather than losing the feature.
        if r.status_code == 400 and "reasoning_effort" in r.text:
            logger.warning("Endpoint rejected reasoning_effort, retrying without it")
            self._send_reasoning_effort = False
            body.pop("reasoning_effort", None)
            r = self._post(body, timeout)

        if r.status_code == 402:
            raise LLMUnavailable(
                "The AI service is out of credit. Dashboard, feed and emotions "
                "are unaffected."
            )
        if r.status_code == 429:
            raise LLMUnavailable("AI rate limit reached. Try again shortly.")
        if r.status_code != 200:
            logger.error("Cerebras %s: %s", r.status_code, r.text[:300])
            raise LLMUnavailable(f"AI service error ({r.status_code}).")

        data = r.json()
        usage = data.get("usage") or {}
        logger.info(
            "cerebras usage prompt=%s completion=%s",
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError):
            logger.error("Unexpected Cerebras response: %s", str(data)[:300])
            return ""


@st.cache_resource
def get_cerebras_llm():
    """Initialize the Cerebras client."""
    api_key = os.getenv("CEREBRAS_API_KEY")
    if not api_key:
        logger.warning("CEREBRAS_API_KEY not found")
        return None
    return CerebrasLLM(api_key=api_key)
