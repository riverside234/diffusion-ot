"""Persistent diagnostics for offline fits/tests, including unsuccessful attempts."""
from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import sys
import time
import traceback
import uuid

from .feature_bank import write_json


def _json_value(value):
    if isinstance(value, Path):
        return str(value.resolve())
    raise TypeError(f"Cannot serialize log value {type(value).__name__}")


class _Tee:
    def __init__(self, original, handle):
        self.original, self.handle = original, handle

    def write(self, text):
        self.original.write(text)
        self.handle.write(text)
        self.handle.flush()
        return len(text)

    def flush(self):
        self.original.flush()
        self.handle.flush()

    def __getattr__(self, name):
        return getattr(self.original, name)


class RunLog:
    """One log directory per attempt; data artifacts remain immutable on resume.

    Output is flushed per write. As with ordinary Python logging, this cannot
    record a hard process kill, but the last flushed event and 'running' status
    remain available. This context controls process stdout/stderr: use serially.
    """
    def __init__(self, directory, operation, metadata=None):
        self.run_id = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}"
        self.directory = Path(directory) / "logs" / self.run_id
        self.metadata = json.loads(json.dumps(metadata or {}, default=_json_value, allow_nan=False))
        self.operation = operation

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=False)
        self.start = time.perf_counter()
        self.stack = ExitStack()
        stdout = self.stack.enter_context((self.directory / "stdout.log").open("a", encoding="utf-8"))
        stderr = self.stack.enter_context((self.directory / "stderr.log").open("a", encoding="utf-8"))
        self.stack.enter_context(redirect_stdout(_Tee(sys.stdout, stdout)))
        self.stack.enter_context(redirect_stderr(_Tee(sys.stderr, stderr)))
        self.report = dict(schema="infoot_run_log_v1", run_id=self.run_id, operation=self.operation,
            status="running", started_at=datetime.now(timezone.utc).isoformat(), metadata=self.metadata,
            python=platform.python_version(), platform=platform.platform(), command=sys.argv,
            working_directory=str(Path.cwd()))
        self.write_json("run.json", self.report)
        self.event("run_started", operation=self.operation)
        print(f"Saved run logs: {self.directory}", flush=True)
        return self

    def append_jsonl(self, filename, payload):
        with (self.directory / filename).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, default=_json_value, allow_nan=False) + "\n")

    def event(self, event, **fields):
        self.append_jsonl("events.jsonl", dict(event=event, timestamp=datetime.now(timezone.utc).isoformat(),
            elapsed_seconds=time.perf_counter() - self.start, **fields))

    def write_json(self, filename, payload):
        write_json(self.directory / filename, json.loads(json.dumps(payload, default=_json_value, allow_nan=False)))

    def __exit__(self, exc_type, error, tb):
        try:
            status = "completed" if error is None else "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
            self.report.update(status=status, elapsed_seconds=time.perf_counter() - self.start,
                               finished_at=datetime.now(timezone.utc).isoformat())
            if error is not None:
                self.report.update(error_type=exc_type.__name__, error=str(error))
                self.write_json("error.json", dict(error_type=exc_type.__name__, message=str(error)))
                with (self.directory / "traceback.txt").open("w", encoding="utf-8") as handle:
                    traceback.print_exception(exc_type, error, tb, file=handle)
                traceback.print_exception(exc_type, error, tb)
            self.event("run_finished", status=status)
            self.write_json("run.json", self.report)
        finally:
            self.stack.close()
        return False
