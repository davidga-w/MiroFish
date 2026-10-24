"""Client-supplied IDs must never resolve outside their storage directories."""

import pytest

from app import create_app
from app.services.report_agent import ReportManager
from app.utils.path_safety import UnsafePathError, is_safe_id, safe_child_path


@pytest.mark.parametrize("value", [
    "proj_e2ebcc73f471",
    "sim_91fd7440417a",
    "report_0123456789ab",
    "3f2b6c1e-6f1a-4b8e-9d2c-1a2b3c4d5e6f",
])
def test_generated_ids_are_safe(value):
    assert is_safe_id(value)
    assert safe_child_path("/data", value) == f"/data/{value}"


@pytest.mark.parametrize("value", [
    "..", ".", "", "../uploads", "a/b", "a\\b", ".hidden", "sim_1\x00", "sim_1\n",
    None, 7, "x" * 129,
])
def test_traversal_ids_are_rejected(value):
    assert not is_safe_id(value)
    with pytest.raises(UnsafePathError):
        safe_child_path("/data", value)


def test_report_delete_cannot_escape_the_reports_dir(tmp_path, monkeypatch):
    reports_dir = tmp_path / "reports"
    reports_dir.mkdir()
    sibling = tmp_path / "projects"
    sibling.mkdir()
    monkeypatch.setattr(ReportManager, "REPORTS_DIR", str(reports_dir))

    with pytest.raises(UnsafePathError):
        ReportManager.delete_report("..")
    assert ReportManager.get_report("..") is None
    assert sibling.is_dir() and reports_dir.is_dir()


@pytest.fixture(scope="module")
def client():
    return create_app().test_client()


@pytest.mark.parametrize("method, url, body", [
    ("delete", "/api/report/..", None),
    ("delete", "/api/graph/project/..", None),
    ("get", "/api/simulation/../run-status/detail?cursor=", None),
    ("get", "/api/simulation/list?project_id=../projects", None),
    ("post", "/api/simulation/start", {"simulation_id": "../../etc"}),
    ("post", "/api/report/generate", {"simulation_id": "../projects/x"}),
    ("get", "/api/simulation/sim_000000000000/posts?platform=../../x", None),
])
def test_api_rejects_traversal(client, method, url, body):
    response = getattr(client, method)(url, json=body)

    assert response.status_code == 400
    assert response.get_json()["success"] is False


def test_api_still_serves_well_formed_ids(client):
    response = client.get("/api/simulation/sim_000000000000/run-status")

    assert response.status_code == 200
    assert response.get_json()["data"]["runner_status"] == "idle"
