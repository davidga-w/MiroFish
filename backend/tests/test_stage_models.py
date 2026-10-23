import json

from app.config import Config
from app.services import simulation_runner as runner_module
from app.services.report_agent import ReportAgent
from app.services.simulation_runner import SimulationRunner


def test_simulation_subprocess_uses_agent_model(monkeypatch, tmp_path):
    simulation_id = "sim-agent-model"
    sim_dir = tmp_path / "runs" / simulation_id
    scripts_dir = tmp_path / "scripts"
    sim_dir.mkdir(parents=True)
    scripts_dir.mkdir()
    (sim_dir / "simulation_config.json").write_text(
        json.dumps({"time_config": {"total_simulation_hours": 1, "minutes_per_round": 60}}),
        encoding="utf-8",
    )
    (scripts_dir / "run_twitter_simulation.py").write_text("pass\n", encoding="utf-8")

    class Process:
        pid = 123

        def poll(self):
            return None

    class IdleThread:
        def __init__(self, **_kwargs):
            pass

        def start(self):
            pass

    popen_kwargs = {}

    def fake_popen(*_args, **kwargs):
        popen_kwargs.update(kwargs)
        return Process()

    monkeypatch.setattr(Config, "SIMULATION_AGENT_MODEL_NAME", "agent-model")
    monkeypatch.setattr(SimulationRunner, "RUN_STATE_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr(SimulationRunner, "SCRIPTS_DIR", str(scripts_dir))
    monkeypatch.setattr(runner_module.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner_module.threading, "Thread", IdleThread)
    monkeypatch.setattr(
        SimulationRunner,
        "_sync_simulation_status",
        classmethod(lambda _cls, *_args, **_kwargs: None),
    )

    try:
        SimulationRunner.start_simulation(
            simulation_id,
            platform="twitter",
            enable_graph_memory_update=False,
        )
        assert popen_kwargs["env"]["LLM_MODEL_NAME"] == "agent-model"
    finally:
        SimulationRunner._run_states.pop(simulation_id, None)
        SimulationRunner._processes.pop(simulation_id, None)
        SimulationRunner._action_queues.pop(simulation_id, None)
        for files in (SimulationRunner._stdout_files, SimulationRunner._stderr_files):
            handle = files.pop(simulation_id, None)
            if handle:
                handle.close()
        SimulationRunner._graph_memory_enabled.pop(simulation_id, None)


def test_report_agent_uses_report_model(monkeypatch):
    monkeypatch.setattr(Config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(Config, "REPORT_AGENT_MODEL_NAME", "report-model")

    agent = ReportAgent(
        graph_id="graph",
        simulation_id="sim",
        simulation_requirement="requirement",
        zep_tools=object(),
    )

    assert agent.llm.model == "report-model"
