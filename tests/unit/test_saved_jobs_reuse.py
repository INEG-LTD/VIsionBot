from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
import types
import json
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_app_module():
    if "chime" not in sys.modules:
        sys.modules["chime"] = types.SimpleNamespace(
            theme=lambda *_args, **_kwargs: None,
            success=lambda *_args, **_kwargs: None,
            error=lambda *_args, **_kwargs: None,
        )
    return import_module("job_application_app.app")


def test_load_matching_saved_google_jobs_filters_to_requested_search_query() -> None:
    app_module = _load_app_module()
    with TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "google-jobs-list.jsonl"
        rows = [
            {
                "job_title": "Junior .NET Developer",
                "search_query": "junior it developer in england",
            },
            {
                "job_title": "Jobs at Jobster",
                "search_query": "jobs at jobster",
            },
        ]
        with jsonl_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

        matches = app_module._load_matching_saved_google_jobs(
            jsonl_path,
            requested_search_query="junior it developer in england",
        )

    assert len(matches) == 1
    assert matches[0]["job_title"] == "Junior .NET Developer"


def test_load_matching_saved_google_jobs_returns_empty_for_unrelated_saved_rows() -> None:
    app_module = _load_app_module()
    with TemporaryDirectory() as tmpdir:
        jsonl_path = Path(tmpdir) / "google-jobs-list.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"job_title": "Jobs at Jobster", "search_query": "jobs at jobster"}) + "\n")

        matches = app_module._load_matching_saved_google_jobs(
            jsonl_path,
            requested_search_query="junior it developer in england",
        )

    assert matches == []
