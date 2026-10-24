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


@pytest.fixture(autouse=True)
def isolated_upload_dirs(tmp_path_factory, monkeypatch):
    """Keep test projects, simulations and reports out of the real uploads/.

    Looking up a simulation creates its folder, so tests using IDs such as
    "sim-1" would otherwise leave empty folders next to the user's real runs.
    A test can still point one store at its own tmp_path.
    """
    from app.config import Config
    from app.models.project import ProjectManager
    from app.services.report_agent import ReportManager
    from app.services.simulation_manager import SimulationManager
    from app.services.simulation_runner import SimulationRunner

    uploads = tmp_path_factory.mktemp("uploads")
    simulations = str(uploads / "simulations")
    monkeypatch.setattr(Config, "UPLOAD_FOLDER", str(uploads))
    monkeypatch.setattr(Config, "OASIS_SIMULATION_DATA_DIR", simulations)
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", simulations)
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", simulations)
    monkeypatch.setattr(ProjectManager, "PROJECTS_DIR", str(uploads / "projects"))
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(uploads / "reports"))
