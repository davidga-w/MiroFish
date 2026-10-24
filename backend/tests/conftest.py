"""Shared fixtures for the backend test suite."""

import pytest


@pytest.fixture(autouse=True)
def isolated_llm_spend_ledger(tmp_path_factory, monkeypatch):
    """Record test LLM usage in a throwaway ledger.

    The real ledger (uploads/llm_spend.json) enforces the daily and per-run
    spending caps, so fake usage recorded by tests would eat into them.
    """
    ledger_dir = tmp_path_factory.mktemp("llm-spend")
    monkeypatch.setenv("LLM_SPEND_LEDGER_PATH", str(ledger_dir / "llm_spend.json"))
