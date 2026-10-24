"""
Cost guards for OASIS agent LLM requests.

Why this exists: OASIS agents (CAMEL ChatAgents) resend their whole chat history on
every turn, and CAMEL only trims history at the model's context limit, which it looks
up by model name. It does not know claude-haiku-5-5 and falls back to a
999,999,999-token limit, so agent prompts grew to ~740k tokens and crossed Haiku 5.5's
5x price for prompts over 100k tokens. OASIS also serialises every feed with
json.dumps' default ensure_ascii=True, turning non-ASCII text into \\uXXXX escapes
(about twice the tokens).

Settings (environment variables):
    SIMULATION_AGENT_HISTORY_TURNS  default 2       earlier turns an agent keeps
    SIMULATION_MAX_PROMPT_TOKENS    default 40000   hard ceiling per agent request
    SIMULATION_LLM_CONCURRENCY      default 10      parallel agent requests per platform
Spend limits come from app/utils/llm_spend.py (SIMULATION_SPEND_LIMIT_USD and
LLM_DAILY_SPEND_LIMIT_USD).
"""

import json
import logging
import os
from typing import Any, Callable, Dict, List, Optional

from app.utils.llm_spend import budget_status, check_budget, record_response_usage

logger = logging.getLogger("mirofish.llm_cost_guard")


def _int_env(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except ValueError:
        return default


HISTORY_TURNS = _int_env("SIMULATION_AGENT_HISTORY_TURNS", 2, minimum=0)
MAX_PROMPT_TOKENS = _int_env("SIMULATION_MAX_PROMPT_TOKENS", 40_000, minimum=4_000)
LLM_CONCURRENCY = _int_env("SIMULATION_LLM_CONCURRENCY", 10, minimum=1)

# OpenAI SDK retries honour the provider's retry-after header; CAMEL then retries
# rate-limited turns with jittered exponential backoff (its default gives up in ~2s).
API_MAX_RETRIES = 4
AGENT_RETRY_ATTEMPTS = 5
AGENT_RETRY_DELAY_SECONDS = 8.0

_SYSTEM_ROLES = {"system", "developer"}


def spend_scope_for(config: Dict[str, Any]) -> str:
    return f"simulation:{config.get('simulation_id', 'unknown')}"


def budget_stop_reason(scope: str) -> Optional[str]:
    """Why the simulation must stop spending, or None while within budget."""
    status = budget_status(scope)
    return status["reason"] if status["exhausted"] else None


# ---------------------------------------------------------------- plain-text feeds


class _PlainTextJson:
    """Stand-in for the json module whose dumps() keeps non-ASCII text readable."""

    def __getattr__(self, name: str) -> Any:
        return getattr(json, name)

    @staticmethod
    def dumps(obj: Any, *args: Any, **kwargs: Any) -> str:
        kwargs.setdefault("ensure_ascii", False)
        return json.dumps(obj, *args, **kwargs)


def install_plain_text_feeds() -> None:
    """Send feeds to agents as plain UTF-8 text instead of \\uXXXX escapes."""
    from oasis.social_agent import agent_environment

    if not isinstance(agent_environment.json, _PlainTextJson):
        agent_environment.json = _PlainTextJson()


# ---------------------------------------------------------------- history


def _role(item: Any) -> str:
    role = item.get("role_at_backend", item.get("role")) if isinstance(item, dict) else None
    return str(getattr(role, "value", role) or "")


def _split_turns(items: List[Any]) -> List[List[Any]]:
    """Group a chat history into turns; each turn starts at a user message."""
    turns: List[List[Any]] = []
    for item in items:
        if _role(item) == "user" or not turns:
            turns.append([item])
        else:
            turns[-1].append(item)
    return turns


def _leading_system_count(items: List[Any]) -> int:
    count = 0
    while count < len(items) and _role(items[count]) in _SYSTEM_ROLES:
        count += 1
    return count


def trim_agent_history(agent: Any, keep_turns: int = HISTORY_TURNS) -> int:
    """Keep the system prompt plus the agent's last `keep_turns` answered turns.

    Turns the model never answered (rate-limited or failed) are dropped. Returns how
    many history records were removed.
    """
    storage = getattr(getattr(getattr(agent, "memory", None), "_chat_history_block", None), "storage", None)
    if storage is None:
        logger.warning("Agent memory layout not recognised; history left untrimmed")
        return 0

    records = storage.load()
    head = _leading_system_count(records)
    answered = [
        turn for turn in _split_turns(records[head:])
        if any(_role(item) != "user" for item in turn)
    ]
    kept = answered[-keep_turns:] if keep_turns > 0 else []
    trimmed = records[:head] + [item for turn in kept for item in turn]
    if len(trimmed) == len(records):
        return 0
    storage.clear()
    storage.save(trimmed)
    return len(records) - len(trimmed)


def fit_prompt(messages: List[Any], count_tokens: Callable[[List[Any]], int], limit: int) -> List[Any]:
    """Drop the oldest whole turns until the request fits `limit` tokens.

    The system prompt and the current turn are always kept, so a request is never
    reshaped into something the provider would reject.
    """
    if count_tokens(messages) <= limit:
        return messages
    head = _leading_system_count(messages)
    turns = _split_turns(messages[head:])
    while len(turns) > 1 and count_tokens(messages[:head] + [m for t in turns for m in t]) > limit:
        turns.pop(0)
    fitted = messages[:head] + [m for t in turns for m in t]
    logger.warning("Agent request trimmed to fit %s tokens (%s -> %s messages)", limit, len(messages), len(fitted))
    return fitted


# ---------------------------------------------------------------- model wrapper


def _token_counter(model: Any) -> Callable[[List[Any]], int]:
    counter = getattr(model, "token_counter", None)

    def count(messages: List[Any]) -> int:
        try:
            return counter.count_tokens_from_messages(messages)
        except Exception:
            # One token per character over-counts most text: the safe direction here.
            return len(json.dumps(messages, ensure_ascii=False, default=str))

    return count


def guard_model(model: Any, scope: str, model_name: str) -> Any:
    """Make every request through `model` budget-checked, size-capped and recorded."""
    count_tokens = _token_counter(model)
    original_arun = model.arun
    original_run = model.run

    def prepare(messages: List[Any]) -> List[Any]:
        check_budget(scope)  # raises LLMBudgetExceeded before anything is sent
        return fit_prompt(messages, count_tokens, MAX_PROMPT_TOKENS)

    async def guarded_arun(messages: List[Any], *args: Any, **kwargs: Any) -> Any:
        messages = prepare(messages)
        response = await original_arun(messages, *args, **kwargs)
        record_response_usage(model_name, response, messages, scope=scope)
        return response

    def guarded_run(messages: List[Any], *args: Any, **kwargs: Any) -> Any:
        messages = prepare(messages)
        response = original_run(messages, *args, **kwargs)
        record_response_usage(model_name, response, messages, scope=scope)
        return response

    model.arun = guarded_arun
    model.run = guarded_run
    return model


def configure_agent_retries(agent_graph: Any) -> None:
    """Give rate-limited agent turns real backoff instead of CAMEL's ~2s of retries."""
    for _, agent in agent_graph.get_agents():
        agent.retry_attempts = max(int(getattr(agent, "retry_attempts", 1)), AGENT_RETRY_ATTEMPTS)
        agent.retry_delay = max(float(getattr(agent, "retry_delay", 0.0)), AGENT_RETRY_DELAY_SECONDS)
