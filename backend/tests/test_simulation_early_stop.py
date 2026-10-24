"""A clean exit is "completed" only when every platform finished all its rounds."""

import json

import pytest

from app.api.report import _run_is_reportable
from app.services.simulation_runner import RunnerStatus, SimulationRunState, SimulationRunner
from app.utils.locale import get_locale


class ExitedProcess:
    pid = 4242
    returncode = 0

    def poll(self):
        return 0


@pytest.fixture
def run_monitor(tmp_path, monkeypatch):
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(
        SimulationRunner,
        "_sync_simulation_status",
        classmethod(lambda _cls, *_args, **_kwargs: None),
    )

    def run(twitter_log):
        simulation_id = "sim_early"
        log = tmp_path / simulation_id / "twitter" / "actions.jsonl"
        log.parent.mkdir(parents=True)
        log.write_text("".join(json.dumps(entry) + "\n" for entry in twitter_log), encoding="utf-8")
        monkeypatch.setitem(SimulationRunner._run_states, simulation_id, SimulationRunState(
            simulation_id=simulation_id,
            runner_status=RunnerStatus.RUNNING,
            total_rounds=96,
            twitter_running=True,
        ))
        monkeypatch.setitem(SimulationRunner._processes, simulation_id, ExitedProcess())
        SimulationRunner._graph_memory_enabled.pop(simulation_id, None)

        SimulationRunner._monitor_simulation(simulation_id, get_locale())
        return SimulationRunner.get_run_state(simulation_id)

    return run


def _round_end(round_num):
    return {"event_type": "round_end", "round": round_num, "simulated_hours": round_num}


def test_full_run_completes(run_monitor):
    state = run_monitor([
        _round_end(96),
        {"event_type": "simulation_end", "total_rounds": 96, "total_actions": 0},
    ])

    assert state.runner_status == RunnerStatus.COMPLETED
    assert state.error is None and state.stop_reason is None


def test_spending_cap_stop_is_reported_as_stopped_early(run_monitor):
    state = run_monitor([
        _round_end(40),
        {"event_type": "simulation_end", "total_rounds": 40, "total_actions": 0, "stop_reason": "budget"},
    ])

    assert state.runner_status == RunnerStatus.STOPPED
    assert state.stop_reason == "budget"
    assert state.error == "Stopped early at round 40/96: LLM spending limit reached"
    assert state.to_dict()["stop_reason"] == "budget"


def test_clean_exit_without_simulation_end_fails(run_monitor):
    state = run_monitor([_round_end(12)])

    assert state.runner_status == RunnerStatus.FAILED
    assert "round 12/96" in state.error


def test_incomplete_graph_opt_in_still_needs_every_round(tmp_path, monkeypatch):
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    (tmp_path / "sim_f" / "twitter").mkdir(parents=True)
    (tmp_path / "sim_f" / "twitter" / "actions.jsonl").write_text("", encoding="utf-8")

    def failed_run(stop_reason):
        return SimulationRunState(
            simulation_id="sim_f",
            runner_status=RunnerStatus.FAILED,
            twitter_completed=True,
            stop_reason=stop_reason,
        )

    assert _run_is_reportable(failed_run(None), accept_incomplete_graph=True)
    assert not _run_is_reportable(failed_run("budget"), accept_incomplete_graph=True)
