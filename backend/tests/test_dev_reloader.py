"""Saving code must not restart the debug server under in-flight work."""

import pytest

from app.models.task import TaskManager, TaskStatus
from app.utils import dev_reloader


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(dev_reloader, "_POLL_SECONDS", 0)


def test_reload_waits_until_nothing_is_in_flight(monkeypatch):
    checks = iter([["simulation sim_1"], ["simulation sim_1"], []])
    calls = []
    monkeypatch.setattr(dev_reloader, "_busy_work", lambda: calls.append(1) or next(checks))

    with pytest.raises(SystemExit) as exit_info:
        dev_reloader.DeferredReloaderLoop().trigger_reload("app/api/simulation.py")

    assert exit_info.value.code == 3  # Werkzeug's "restart me" exit code
    assert len(calls) == 3


def test_idle_reload_is_immediate(monkeypatch):
    monkeypatch.setattr(dev_reloader, "_busy_work", lambda: [])

    with pytest.raises(SystemExit) as exit_info:
        dev_reloader.DeferredReloaderLoop().trigger_reload("run.py")

    assert exit_info.value.code == 3


def test_running_background_task_counts_as_busy():
    manager = TaskManager()
    task_id = manager.create_task("report_generate")
    manager.update_task(task_id, status=TaskStatus.PROCESSING)
    try:
        assert f"report_generate task {task_id}" in dev_reloader._busy_work()
    finally:
        manager.complete_task(task_id, {})

    assert not any(task_id in item for item in dev_reloader._busy_work())


def test_install_registers_the_loop_with_werkzeug():
    from werkzeug._reloader import reloader_loops

    assert reloader_loops[dev_reloader.install_deferred_reloader()] is dev_reloader.DeferredReloaderLoop
