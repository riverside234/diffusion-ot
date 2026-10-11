"""Coarse fitting-phase timers; no synchronization inside numerical loops."""
from contextlib import contextmanager
import time

import torch


class PairTimer:
    def __init__(self, device):
        self.device = torch.device(device)
        self.host = {}
        self.events = {}

    @contextmanager
    def section(self, name, *, cuda=True):
        use_cuda = cuda and self.device.type == "cuda"
        if use_cuda:
            stream = torch.cuda.current_stream(self.device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record(stream)
        started = time.perf_counter()
        try:
            yield
        finally:
            if use_cuda:
                end.record(stream)
                self.events.setdefault(name, []).append((start, end))
            self.host[name] = self.host.get(name, 0.) + time.perf_counter() - started

    def report(self):
        # Called only at batch completion/interruption, never per inner step.
        if self.events:
            torch.cuda.synchronize(self.device)
        return dict(host_wall_seconds=dict(self.host),
            cuda_stream_seconds={name: sum(a.elapsed_time(b) for a, b in events) / 1000
                                 for name, events in self.events.items()},
            note="Host times include waits. CUDA stream intervals may include host launch gaps; do not add host and CUDA times. Checkpoint/logging includes device-to-host copy, CPU validation and file I/O.")
