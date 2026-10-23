from app import create_app
from app.config import Config
from app.models.task import TaskManager, TaskStatus
from app.services.simulation_manager import (
    SimulationManager,
    SimulationState,
    SimulationStatus,
)


def _client(tmp_path, monkeypatch):
    monkeypatch.setattr(SimulationManager, "SIMULATION_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(Config, "OASIS_SIMULATION_DATA_DIR", str(tmp_path))
    app = create_app()
    app.config.update(TESTING=True)
    return app.test_client()


def _save_preparing_state(simulation_id):
    SimulationManager()._save_simulation_state(
        SimulationState(
            simulation_id=simulation_id,
            project_id="proj_missing_for_prepare_test",
            graph_id="graph",
            status=SimulationStatus.PREPARING,
            entities_count=7,
        )
    )


def test_prepare_reuses_in_flight_task(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    _save_preparing_state("sim_prepare_active")
    task_id = TaskManager().create_task(
        "simulation_prepare", metadata={"simulation_id": "sim_prepare_active"}
    )

    try:
        response = client.post(
            "/api/simulation/prepare", json={"simulation_id": "sim_prepare_active"}
        )
    finally:
        TaskManager().update_task(task_id, status=TaskStatus.COMPLETED)

    assert response.status_code == 200
    data = response.json["data"]
    assert data["task_id"] == task_id
    assert data["reused"] is True
    assert data["already_prepared"] is False
    assert data["expected_entities_count"] == 7


def test_prepare_ignores_finished_task(tmp_path, monkeypatch):
    client = _client(tmp_path, monkeypatch)
    _save_preparing_state("sim_prepare_finished")
    task_id = TaskManager().create_task(
        "simulation_prepare", metadata={"simulation_id": "sim_prepare_finished"}
    )
    TaskManager().update_task(task_id, status=TaskStatus.FAILED)

    response = client.post(
        "/api/simulation/prepare", json={"simulation_id": "sim_prepare_finished"}
    )

    # With no in-flight task the request takes the normal path, which stops
    # here because the test simulation points at a project that does not exist.
    assert response.status_code == 404
    assert "data" not in response.json
