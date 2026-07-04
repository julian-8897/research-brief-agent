from __future__ import annotations

import json
import logging

import pytest

from src.observability.run_records import RunRecordStore


def _read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_disabled_store_is_noop():
    store = RunRecordStore(None)
    assert store.enabled is False
    # Must not raise even though there is nowhere to write.
    store.start("run", {"a": 1})
    store.event("run", {"b": 2})
    store.finish("run", {"c": 3})


def test_best_effort_write_persists_records(tmp_path):
    store = RunRecordStore(str(tmp_path))
    store.start("run", {"request_id": "req"})
    store.event("run", {"event": {"event": "final"}})
    store.finish("run", {"status": "completed"})

    rows = _read(tmp_path / "run.jsonl")
    assert [row["type"] for row in rows] == ["run_started", "event", "run_finished"]
    index = _read(tmp_path / "runs.jsonl")
    assert [row["type"] for row in index] == ["run_started", "run_finished"]


def test_best_effort_swallows_io_errors_and_logs_once(tmp_path, caplog):
    # A file where the store expects its directory forces mkdir/open to fail.
    collision = tmp_path / "recs"
    collision.write_text("", encoding="utf-8")
    store = RunRecordStore(str(collision))

    with caplog.at_level(logging.INFO):
        # None of these should propagate the underlying OSError.
        store.start("run", {"request_id": "req"})
        store.event("run", {"event": {"event": "final"}})
        store.finish("run", {"status": "completed"})

    failures = [r for r in caplog.records if "run_record_write_failed" in r.getMessage()]
    assert len(failures) == 1


def test_required_store_propagates_write_errors(tmp_path):
    directory = tmp_path / "recs"
    directory.mkdir()
    store = RunRecordStore(str(directory), required=True)
    # Replace the run directory with a file so the next write collides.
    store.directory = tmp_path / "collision"
    store.directory.write_text("", encoding="utf-8")

    with pytest.raises(OSError):
        store.start("run", {"request_id": "req"})


def test_required_store_verifies_writability_at_construction(tmp_path):
    collision = tmp_path / "recs"
    collision.write_text("", encoding="utf-8")

    with pytest.raises(OSError):
        RunRecordStore(str(collision), required=True)


def test_serialization_errors_surface_even_when_best_effort(tmp_path):
    store = RunRecordStore(str(tmp_path))

    class Unserializable:
        def __repr__(self):  # str fallback used by json default=str
            raise ValueError("boom")

    with pytest.raises(ValueError):
        store.event("run", {"event": Unserializable()})
