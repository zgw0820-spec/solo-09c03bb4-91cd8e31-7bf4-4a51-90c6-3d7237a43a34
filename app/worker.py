"""Background training worker.

Both worker containers run this process with a different ``WORKER_ID``.  Each
worker registers itself, publishes a heartbeat, polls for an unfinished job
whose lease is free or expired, and then trains one batch at a time.  Every
batch commit writes the full checkpoint (weights, momentum, batch loss, dataset
loss, step, timestamp, worker and lease epoch) plus the job's progress in one
transaction.

Slow-batch liveness
-------------------

A batch may take longer than ``TRAINING_LEASE_SECONDS`` (real training work,
not a dead process).  Renewing the lease only *between* batches would let the
other worker steal it mid-batch; the fenced commit would then fail and the two
workers would take over from each other forever with the job stuck at step 0
(a lease livelock).

To prevent that, every worker runs a daemon :class:`LeaseWatchdog` thread that
renews the lease and refreshes the heartbeat continuously for as long as the
worker holds a lease and is actively training - including while a batch is
being computed.  The main training loop suspends the watchdog on purpose during
the inter-batch demo pause and the watchdog stops entirely when the process
exits, so a frozen or dead worker still stops renewing, its lease still
expires, and another instance can still take over.

If the process dies, its heartbeat stops; the other worker waits for the lease
to expire and resumes from the newest committed step.  If a worker is merely
suspended and wakes up later, its committed fence check fails and it cannot
write, so it can neither roll the breakpoint back nor report a false completion.
"""
from __future__ import annotations

import argparse
import os
import signal
import socket
import sys
import threading
import time
import uuid
from pathlib import Path

from . import lease as lease_mod
from . import store
from .config import Settings
from .db import init_db
from .lease import Lease, LeaseLost


class LeaseWatchdog:
    """Background lease renewal that keeps a slow batch alive.

    While ``active`` is set (the worker owns a lease and is training), the
    thread renews the fenced lease roughly once per heartbeat interval, even
    while the main thread is blocked inside a long batch.  Renewal is a no-op
    while inactive, which is what lets the lease expire during an inter-batch
    pause or after the process exits.
    """

    def __init__(
        self,
        settings: Settings,
        worker_id: str,
        stop_event: threading.Event,
    ) -> None:
        self.settings = settings
        self.worker_id = worker_id
        self.stop_event = stop_event
        self._lease: Lease | None = None
        self._active = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="lease-watchdog", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def attach(self, lease: Lease) -> None:
        """Start protecting ``lease`` while batches are processed."""
        with self._lock:
            self._lease = lease
        self._active.set()

    def suspend(self) -> None:
        """Stop renewing (e.g. inter-batch pause) so the lease may expire."""
        self._active.clear()

    def resume(self) -> None:
        """Resume renewing after a suspension."""
        self._active.set()

    def detach(self) -> None:
        """Stop protecting the current lease entirely."""
        self._active.clear()
        with self._lock:
            self._lease = None

    def _run(self) -> None:
        while not self.stop_event.is_set():
            if self.stop_event.wait(self.settings.heartbeat_interval):
                return
            if not self._active.is_set():
                continue
            with self._lock:
                lease = self._lease
            if lease is None:
                continue
            try:
                ok = lease_mod.renew_with_heartbeat(
                    self.settings.db_path, lease, self.worker_id, "training"
                )
            except Exception as exc:  # pragma: no cover - DB hiccup, retry next tick
                print(f"[{self.worker_id}] lease renewal error: {exc!r}", flush=True)
                continue
            if not ok:
                # Another instance took over.  The main thread finds out either
                # from its own pre-batch renewal or from the commit-time fence;
                # there is nothing safe for this thread to write.
                print(
                    f"[{self.worker_id}] watchdog noticed lease epoch "
                    f"{lease.epoch} moved on",
                    flush=True,
                )
                self._active.clear()


class Worker:
    def __init__(
        self,
        settings: Settings,
        worker_id: str,
        pause_file: Path | str | None = None,
    ) -> None:
        self.settings = settings
        self.worker_id = worker_id
        # Explicit injection point for the inter-batch pause hook; falls back to
        # the environment variable used by the demo and integration tests.
        self.pause_file = pause_file
        self.stop_event = threading.Event()
        self.lease: Lease | None = None
        self._last_heartbeat = 0.0
        self._last_status = ""
        self.watchdog = LeaseWatchdog(settings, worker_id, self.stop_event)

    def request_stop(self, *_: object) -> None:
        self.stop_event.set()

    def beat(self, status: str) -> bool:
        """Publish liveness and renew the lease.

        Returns ``False`` when the lease has been taken over, which the training
        loop treats as a hard stop.  During training the watchdog thread also
        renews continuously; this call additionally updates the worker status
        immediately before each batch.
        """
        now = time.monotonic()
        heartbeat_due = (
            status != self._last_status
            or now - self._last_heartbeat >= self.settings.heartbeat_interval
        )
        if heartbeat_due:
            lease_mod.heartbeat(self.settings.db_path, self.worker_id, status)
            self._last_heartbeat = now
            self._last_status = status
        if self.lease is None:
            return True
        return lease_mod.renew(self.settings.db_path, self.lease, self.settings.lease_seconds)

    def run(self) -> None:
        self.settings.validate()
        init_db(self.settings.db_path)
        lease_mod.register_worker(self.settings.db_path, self.worker_id)
        self.beat("idle")
        self.watchdog.start()
        print(f"[{self.worker_id}] started", flush=True)
        try:
            while not self.stop_event.is_set():
                if not self._run_one_job():
                    self.beat("idle")
                    self.stop_event.wait(self.settings.poll_interval)
                    continue
                try:
                    self._train()
                finally:
                    self.watchdog.detach()
                    if self.lease is not None:
                        lease_mod.release(self.settings.db_path, self.lease)
                        self.lease = None
        finally:
            self.watchdog.detach()
            if self.lease is not None:
                lease_mod.release(self.settings.db_path, self.lease)
            lease_mod.heartbeat(self.settings.db_path, self.worker_id, "stopped")
            print(f"[{self.worker_id}] stopped", flush=True)

    def _run_one_job(self) -> bool:
        """Claim one job's lease. Returns ``False`` when nothing is claimable."""
        for job_id in store.claimable_jobs(self.settings.db_path):
            lease = lease_mod.acquire(
                self.settings.db_path,
                job_id,
                self.worker_id,
                self.settings.lease_seconds,
            )
            if lease is not None:
                self.lease = lease
                self.watchdog.attach(lease)
                return True
        return False

    def _train(self) -> None:
        assert self.lease is not None
        job_id = self.lease.job_id
        job = store.get_job(self.settings.db_path, job_id)
        if job is None:
            return
        store.record_event(
            self.settings.db_path,
            job_id,
            "training_started",
            {
                "worker_id": self.worker_id,
                "lease_epoch": self.lease.epoch,
                "resumed_from_step": job["completed_steps"],
            },
        )
        print(
            f"[{self.worker_id}] training {job_id} from step "
            f"{job['completed_steps']}/{job['total_steps']} (lease epoch {self.lease.epoch})",
            flush=True,
        )
        while not self.stop_event.is_set():
            try:
                # The watchdog keeps the lease alive even when this batch takes
                # longer than lease_seconds.  Renew first anyway: if another
                # worker took over while we were suspended, leave without
                # touching the job.  The commit below is independently fenced
                # by the lease epoch.
                self.watchdog.resume()
                if not self.beat("training"):
                    raise LeaseLost("lease was taken over")
                result = store.advance_job(
                    self.settings.db_path,
                    self.lease,
                    self.settings.batch_delay_seconds,
                )
            except LeaseLost:
                print(f"[{self.worker_id}] lost lease on {job_id}; stopping", flush=True)
                self.watchdog.suspend()
                store.record_event(
                    self.settings.db_path,
                    job_id,
                    "lease_lost",
                    {"worker_id": self.worker_id, "lease_epoch": self.lease.epoch},
                )
                self.lease = None
                self.watchdog.detach()
                return
            print(
                f"[{self.worker_id}] {job_id} step {result['step']}/{result['total_steps']} "
                f"loss={result['loss']:.6f}",
                flush=True,
            )
            if result["status"] == "completed":
                print(f"[{self.worker_id}] completed {job_id}", flush=True)
                return
            self._pause_after_batch(job_id)
            if self.stop_event.is_set():
                return
            # Coming back from a pause that allowed a takeover must fail fast
            # instead of attempting a fenced commit we know cannot succeed.
            if not lease_mod.renew(
                self.settings.db_path, self.lease, self.settings.lease_seconds
            ):
                print(
                    f"[{self.worker_id}] lease expired while paused on {job_id}; stopping",
                    flush=True,
                )
                store.record_event(
                    self.settings.db_path,
                    job_id,
                    "lease_lost",
                    {"worker_id": self.worker_id, "lease_epoch": self.lease.epoch},
                )
                self.lease = None
                self.watchdog.detach()
                return

    def _pause_after_batch(self, job_id: str) -> None:
        """Simulated freezes for the failover demo; disabled by default.

        ``TRAINING_DEMO_PAUSE_FILE`` (a path shared with the demo) pauses this
        worker between batches, outside any SQLite transaction, without
        heartbeating - so its lease expires and the other worker takes over.

        With ``TRAINING_DEMO_CRASH_AFTER`` set to a step number, the process
        exits hard right after that step commits, modelling a container crash.
        Both are no-ops unless the environment variables are set.
        """
        crash_after = os.environ.get("TRAINING_DEMO_CRASH_AFTER")
        if crash_after and int(crash_after) == int(
            store.get_job(self.settings.db_path, job_id)["completed_steps"]
        ):
            print(f"[{self.worker_id}] simulated crash after step {crash_after}", flush=True)
            os._exit(70)

        pause_file = self.pause_file or os.environ.get("TRAINING_DEMO_PAUSE_FILE")
        if not pause_file:
            return
        path = Path(pause_file)
        if path.exists():
            print(f"[{self.worker_id}] paused by {path}", flush=True)
            # No heartbeat while paused: suspend the watchdog too, so the lease
            # expires and the other worker can take over.
            self.watchdog.suspend()
            while path.exists() and not self.stop_event.is_set():
                self.stop_event.wait(self.settings.poll_interval)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Training worker")
    parser.add_argument("--worker-id", default=None)
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    worker_id = args.worker_id or f"worker-{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
    worker = Worker(settings, worker_id)
    signal.signal(signal.SIGTERM, worker.request_stop)
    signal.signal(signal.SIGINT, worker.request_stop)
    worker.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
