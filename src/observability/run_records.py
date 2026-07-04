from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Any

from src.observability.logging import log_event

logger = logging.getLogger(__name__)


class RunRecordStore:
    """Append-only JSONL records for streamed brief runs.

    Persistence is best-effort by default: operational IO failures (an
    unwritable ``directory``, a full disk) are logged once but never
    propagated, so observability cannot break the agent request/stream path.
    Set ``required=True`` to instead treat persistence as critical -- the
    directory is probed for writability at construction (fail fast at startup)
    and later write failures propagate to the caller.
    """

    def __init__(self, directory: str | None, *, required: bool = False):
        self.directory = Path(directory) if directory else None
        self.required = required
        self._lock = Lock()
        self._logged_failure = False
        if self.required and self.directory is not None:
            self._verify_writable()

    @property
    def enabled(self) -> bool:
        return self.directory is not None

    def start(self, run_id: str, payload: dict[str, Any]) -> None:
        self._append(run_id, {"type": "run_started", **payload})
        self._append_index({"type": "run_started", "run_id": run_id, **payload})

    def event(self, run_id: str, payload: dict[str, Any]) -> None:
        self._append(run_id, {"type": "event", **payload})

    def finish(self, run_id: str, payload: dict[str, Any]) -> None:
        self._append(run_id, {"type": "run_finished", **payload})
        self._append_index({"type": "run_finished", "run_id": run_id, **payload})

    def path_for(self, run_id: str) -> Path:
        if self.directory is None:
            raise RuntimeError("run record store is disabled")
        return self.directory / f"{run_id}.jsonl"

    def _verify_writable(self) -> None:
        """Probe the directory so a misconfigured required store fails at
        startup rather than mid-stream on the first write."""
        assert self.directory is not None
        self.directory.mkdir(parents=True, exist_ok=True)
        probe = self.directory / ".write-probe"
        probe.write_text("", encoding="utf-8")
        probe.unlink()

    def _append(self, run_id: str, payload: dict[str, Any]) -> None:
        if self.directory is None:
            return
        self._append_line(self.path_for(run_id), payload)

    def _append_index(self, payload: dict[str, Any]) -> None:
        if self.directory is None:
            return
        self._append_line(self.directory / "runs.jsonl", payload)

    def _append_line(self, path: Path, payload: dict[str, Any]) -> None:
        row = {"ts": datetime.now(UTC).isoformat(), **payload}
        # Serialize outside the IO guard: a serialization failure is a caller
        # bug, not an operational IO fault, and should surface rather than be
        # silently swallowed as a best-effort write.
        line = json.dumps(row, default=str, sort_keys=True)
        try:
            with self._lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except OSError as exc:
            if self.required:
                raise
            # Log the first failure only; a full disk or unwritable mount would
            # otherwise emit one error per streamed event.
            if not self._logged_failure:
                self._logged_failure = True
                log_event(
                    logger,
                    "run_record_write_failed",
                    path=str(path),
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
