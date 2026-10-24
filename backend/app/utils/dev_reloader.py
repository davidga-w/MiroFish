"""
Debug-mode code reloading that waits for in-flight work.

Werkzeug's reloader restarts the server as soon as a .py file is saved, and the
restart stops every running simulation (the atexit cleanup) and kills background
tasks such as report generation. This loop changes only *when* the restart
happens: it waits until no simulation process, graph-ingestion drain or
background task is in flight, then restarts exactly like the stock stat reloader.
"""

import time

from werkzeug._reloader import StatReloaderLoop, reloader_loops

from .logger import get_logger

logger = get_logger('mirofish.reloader')

RELOADER_TYPE = "deferred"
_POLL_SECONDS = 5


def _busy_work() -> list:
    """Descriptions of the work a restart would interrupt right now."""
    from ..models.task import TaskManager, TaskStatus
    from ..services.simulation_runner import RunnerStatus, SimulationRunner
    from ..services.zep_graph_memory_updater import ZepGraphMemoryManager

    active = {
        RunnerStatus.STARTING,
        RunnerStatus.RUNNING,
        RunnerStatus.PAUSED,
        RunnerStatus.STOPPING,
    }
    simulations = set(SimulationRunner.get_running_simulations())
    for simulation_id in ZepGraphMemoryManager.get_simulation_ids():
        # Starting up or draining graph ingestion; a failed updater kept only
        # for a retry is not in flight.
        state = SimulationRunner.get_run_state(simulation_id)
        if state is not None and state.runner_status in active:
            simulations.add(simulation_id)

    busy = [f"simulation {simulation_id}" for simulation_id in sorted(simulations)]
    busy += [
        f"{task['task_type']} task {task['task_id']}"
        for task in TaskManager().list_tasks()
        if task["status"] in (TaskStatus.PENDING.value, TaskStatus.PROCESSING.value)
    ]
    return busy


class DeferredReloaderLoop(StatReloaderLoop):
    """Stat reloader whose restart waits until it would interrupt nothing."""

    name = "deferred stat"

    def trigger_reload(self, filename: str) -> None:
        reported = None
        while True:
            try:
                busy = _busy_work()
            except Exception as error:  # never restart blind into running work
                busy = [f"activity check failed: {error}"]
            if not busy:
                break
            if busy != reported:
                logger.warning(
                    "Code change in %s: reload deferred until idle (%s)",
                    filename,
                    "; ".join(busy),
                )
                reported = busy
            time.sleep(_POLL_SECONDS)
        if reported:
            logger.info("Idle again: applying the deferred reload")
        super().trigger_reload(filename)


def install_deferred_reloader() -> str:
    """Register the loop with Werkzeug and return its `reloader_type` name."""
    reloader_loops[RELOADER_TYPE] = DeferredReloaderLoop
    return RELOADER_TYPE
