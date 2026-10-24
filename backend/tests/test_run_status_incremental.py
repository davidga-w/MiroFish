"""Run-status pollers download each agent action once, via log cursors."""

import json

import pytest

from app.services.simulation_runner import RunnerStatus, SimulationRunState, SimulationRunner


def _action(agent_id, timestamp):
    return {
        "round": 1,
        "timestamp": timestamp,
        "agent_id": agent_id,
        "agent_name": f"agent {agent_id}",
        "action_type": "CREATE_POST",
        "action_args": {},
    }


def _write(path, *records, tail="", mode="a"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode, encoding="utf-8") as f:
        f.writelines(json.dumps(record) + "\n" for record in records)
        f.write(tail)


def _poll(cursor):
    actions, cursor, reset = SimulationRunner.read_new_actions(
        "sim_1", SimulationRunner.parse_actions_cursor(cursor)
    )
    return [(action.platform, action.agent_id) for action in actions], cursor, reset


@pytest.fixture
def logs(tmp_path, monkeypatch):
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path))
    monkeypatch.setitem(SimulationRunner._run_states, "sim_1", SimulationRunState(
        simulation_id="sim_1",
        runner_status=RunnerStatus.RUNNING,
        started_at="2026-10-24T10:00:00",
    ))
    run_dir = tmp_path / "sim_1"
    return run_dir / "twitter" / "actions.jsonl", run_dir / "reddit" / "actions.jsonl"


def test_each_action_is_sent_once_in_time_order(logs):
    twitter, reddit = logs
    _write(twitter, {"event_type": "round_start", "round": 1}, _action(1, "2026-10-24T10:00:02"))
    # The writer is still in the middle of the next line
    _write(reddit, _action(2, "2026-10-24T10:00:01"), tail='{"round": 1, "agent')

    actions, cursor, reset = _poll("")
    assert actions == [("reddit", 2), ("twitter", 1)]
    assert reset is False
    assert _poll(cursor) == ([], cursor, False)

    _write(reddit, tail='_id": 3, "timestamp": "2026-10-24T10:00:03"}\n')
    assert _poll(cursor)[0] == [("reddit", 3)]


def test_restarted_run_starts_the_timeline_over(logs):
    twitter, _ = logs
    _write(twitter, _action(1, "2026-10-24T10:00:01"), _action(2, "2026-10-24T10:00:02"))
    _, cursor, _ = _poll("")

    # A forced restart recreates the log, here with lines of the very same length
    SimulationRunner._run_states["sim_1"].started_at = "2026-10-24T11:00:00"
    _write(
        twitter,
        _action(7, "2026-10-24T11:00:01"),
        _action(8, "2026-10-24T11:00:02"),
        _action(9, "2026-10-24T11:00:03"),
        mode="w",
    )

    actions, _, reset = _poll(cursor)
    assert reset is True
    assert actions == [("twitter", 7), ("twitter", 8), ("twitter", 9)]


def test_truncated_log_starts_the_timeline_over(logs):
    twitter, _ = logs
    _write(twitter, _action(1, "2026-10-24T10:00:01"), _action(2, "2026-10-24T10:00:02"))
    _, cursor, _ = _poll("")

    _write(twitter, _action(5, "2026-10-24T10:00:05"), mode="w")

    assert _poll(cursor)[0::2] == ([("twitter", 5)], True)


@pytest.mark.parametrize("cursor", ["|twitter:abc", "|twitter:-5"])
def test_malformed_cursor_is_rejected(cursor):
    with pytest.raises(ValueError):
        SimulationRunner.parse_actions_cursor(cursor)
