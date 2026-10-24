"""
Persistent LLM spend tracking with hard budgets.

Every LLM request the app makes is priced from the token usage the provider reports
and recorded in one JSON ledger shared by the backend and the OASIS simulation
subprocess. Once a budget is used up, new requests are refused before they are sent:

    LLM_DAILY_SPEND_LIMIT_USD   default 30   all LLM requests of the app per UTC day
    SIMULATION_SPEND_LIMIT_USD  default 15   one simulation's agent requests, across restarts
    REPORT_SPEND_LIMIT_USD      default 10   one report's generation

Set a limit to 0 to disable it. The ledger is uploads/llm_spend.json
(override with LLM_SPEND_LEDGER_PATH). Callers attribute spend to a scope such as
"simulation:<id>" or "report:<id>" with set_spend_scope()/spend_scope(); everything
else is recorded under "app" and only counts toward the daily limit.
"""

from __future__ import annotations

import contextlib
import contextvars
import fcntl
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

logger = logging.getLogger("mirofish.llm_spend")

# USD per million tokens, Anthropic list prices (October 2026). Haiku 5.5 bills every
# token of a request at its long-prompt rate once that prompt exceeds 100,000 tokens.
MODEL_PRICES: Dict[str, Dict[str, float]] = {
    "claude-haiku-5-5": {
        "input": 0.10,
        "output": 0.50,
        "long_prompt_tokens": 100_000,
        "long_input": 0.50,
        "long_output": 2.50,
    },
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}
# Models missing above are priced at the highest list price we know of, so a budget
# stops too early rather than too late. Add a model above to price it exactly.
FALLBACK_PRICE: Dict[str, float] = {"input": 10.00, "output": 50.00}

DEFAULT_LIMITS_USD: Dict[str, float] = {
    "LLM_DAILY_SPEND_LIMIT_USD": 30.0,
    "SIMULATION_SPEND_LIMIT_USD": 15.0,
    "REPORT_SPEND_LIMIT_USD": 10.0,
}
SCOPE_LIMIT_ENV: Dict[str, str] = {
    "simulation": "SIMULATION_SPEND_LIMIT_USD",
    "report": "REPORT_SPEND_LIMIT_USD",
}

_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_scope: contextvars.ContextVar[str] = contextvars.ContextVar("llm_spend_scope", default="app")
_warned_models: set = set()


class LLMBudgetExceeded(RuntimeError):
    """Raised instead of sending an LLM request once a spend limit is used up."""


def reraise_if_budget_exceeded(error: BaseException) -> None:
    """Call first in broad `except` blocks so a budget stop is never swallowed by a fallback."""
    if isinstance(error, LLMBudgetExceeded):
        raise error


# ---------------------------------------------------------------- scopes


def set_spend_scope(scope: str) -> contextvars.Token:
    """Attribute this thread's (or task's) LLM requests to `scope`."""
    return _scope.set(scope)


def current_spend_scope() -> str:
    return _scope.get()


@contextlib.contextmanager
def spend_scope(scope: str) -> Iterator[None]:
    token = _scope.set(scope)
    try:
        yield
    finally:
        _scope.reset(token)


# ---------------------------------------------------------------- pricing


def _price_for(model: Optional[str]) -> Dict[str, float]:
    name = (model or "").strip().lower()
    for known in sorted(MODEL_PRICES, key=len, reverse=True):
        if name == known or name.startswith(known + "-") or name.startswith(known + "@"):
            return MODEL_PRICES[known]
    if name not in _warned_models:
        _warned_models.add(name)
        logger.warning(
            "No price listed for model %r; budgeting it at $%s/$%s per million tokens",
            model, FALLBACK_PRICE["input"], FALLBACK_PRICE["output"],
        )
    return FALLBACK_PRICE


def request_cost_usd(model: Optional[str], input_tokens: int, output_tokens: int) -> float:
    price = _price_for(model)
    long_prompt = price.get("long_prompt_tokens")
    if long_prompt and input_tokens > long_prompt:
        rate_in, rate_out = price["long_input"], price["long_output"]
    else:
        rate_in, rate_out = price["input"], price["output"]
    return (input_tokens * rate_in + output_tokens * rate_out) / 1_000_000


def response_usage(response: Any) -> Optional[Tuple[int, int]]:
    """(input_tokens, output_tokens) reported by an OpenAI-compatible response, if any."""
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        return None

    def read(*names: str) -> int:
        for name in names:
            value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
            if isinstance(value, (int, float)):
                return int(value)
        return 0

    return read("prompt_tokens", "input_tokens"), read("completion_tokens", "output_tokens")


def _estimate_tokens(messages: Optional[List[Any]], response: Any) -> Tuple[int, int]:
    # One token per character over-counts most text, which is the safe direction here.
    try:
        prompt_chars = len(json.dumps(messages or [], ensure_ascii=False, default=str))
    except (TypeError, ValueError):
        prompt_chars = len(str(messages or ""))
    return prompt_chars, len(str(response or ""))


# ---------------------------------------------------------------- ledger


def ledger_path() -> str:
    return os.environ.get("LLM_SPEND_LEDGER_PATH") or os.path.join(
        _BACKEND_DIR, "uploads", "llm_spend.json"
    )


@contextlib.contextmanager
def _locked(exclusive: bool) -> Iterator[str]:
    path = ledger_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".lock", "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        try:
            yield path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _read(path: str) -> Dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as error:
        # Never treat an unreadable ledger as "nothing spent".
        raise LLMBudgetExceeded(
            f"LLM spend ledger {path} is unreadable ({error}); refusing LLM requests "
            "until it is repaired or removed"
        ) from error
    if not isinstance(data, dict):
        raise LLMBudgetExceeded(f"LLM spend ledger {path} is malformed; refusing LLM requests")
    data.setdefault("days", {})
    data.setdefault("scopes", {})
    return data


def _write(path: str, data: Dict[str, Any]) -> None:
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path) or ".", prefix=".llm_spend.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.remove(tmp_path)
        raise


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def record_usage(
    model: Optional[str],
    input_tokens: int,
    output_tokens: int,
    scope: Optional[str] = None,
) -> float:
    """Price one request and add it to the ledger. Returns its cost in USD."""
    cost = request_cost_usd(model, input_tokens, output_tokens)
    scope = scope or current_spend_scope()
    day = _today()
    with _locked(exclusive=True) as path:
        data = _read(path)
        data["days"][day] = round(float(data["days"].get(day, 0.0)) + cost, 6)
        entry = data["scopes"].setdefault(
            scope, {"usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0}
        )
        entry["usd"] = round(float(entry.get("usd", 0.0)) + cost, 6)
        entry["calls"] = int(entry.get("calls", 0)) + 1
        entry["input_tokens"] = int(entry.get("input_tokens", 0)) + int(input_tokens)
        entry["output_tokens"] = int(entry.get("output_tokens", 0)) + int(output_tokens)
        entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        _write(path, data)
    return cost


def record_response_usage(
    model: Optional[str],
    response: Any,
    messages: Optional[List[Any]] = None,
    scope: Optional[str] = None,
) -> float:
    usage = response_usage(response)
    if usage is None:
        usage = _estimate_tokens(messages, response)
        logger.warning("LLM response for %s had no usage data; recorded an over-estimate", model)
    return record_usage(model, usage[0], usage[1], scope=scope)


# ---------------------------------------------------------------- budgets


def _limit(env_name: str) -> Optional[float]:
    raw = os.environ.get(env_name)
    value = DEFAULT_LIMITS_USD[env_name]
    if raw not in (None, ""):
        try:
            value = float(raw)
        except ValueError:
            logger.warning("Ignoring invalid %s=%r; using $%s", env_name, raw, value)
    return value if value > 0 else None


def _scope_limit_env(scope: str) -> Optional[str]:
    return SCOPE_LIMIT_ENV.get(scope.split(":", 1)[0])


def budget_status(scope: Optional[str] = None) -> Dict[str, Any]:
    """Spend so far for `scope` and today, the applicable limits, and whether either is used up."""
    scope = scope or current_spend_scope()
    with _locked(exclusive=False) as path:
        data = _read(path)
    entry = data["scopes"].get(scope, {})
    spent = float(entry.get("usd", 0.0))
    day_spent = float(data["days"].get(_today(), 0.0))
    limit_env = _scope_limit_env(scope)
    scope_limit = _limit(limit_env) if limit_env else None
    daily_limit = _limit("LLM_DAILY_SPEND_LIMIT_USD")

    reason = None
    if scope_limit is not None and spent >= scope_limit:
        reason = (
            f"LLM spend limit reached for {scope}: ${spent:.2f} of ${scope_limit:.2f} "
            f"({limit_env}). Raise the limit to continue."
        )
    elif daily_limit is not None and day_spent >= daily_limit:
        reason = (
            f"Daily LLM spend limit reached: ${day_spent:.2f} of ${daily_limit:.2f} today (UTC) "
            "(LLM_DAILY_SPEND_LIMIT_USD). Raise the limit or wait until tomorrow."
        )
    return {
        "scope": scope,
        "usd": round(spent, 4),
        "limit_usd": scope_limit,
        "calls": int(entry.get("calls", 0)),
        "input_tokens": int(entry.get("input_tokens", 0)),
        "output_tokens": int(entry.get("output_tokens", 0)),
        "day_usd": round(day_spent, 4),
        "daily_limit_usd": daily_limit,
        "exhausted": reason is not None,
        "reason": reason,
    }


def check_budget(scope: Optional[str] = None) -> None:
    """Raise LLMBudgetExceeded if the scope's or today's limit is used up."""
    status = budget_status(scope)
    if status["exhausted"]:
        raise LLMBudgetExceeded(status["reason"])
