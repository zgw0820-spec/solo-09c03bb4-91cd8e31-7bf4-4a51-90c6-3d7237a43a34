"""Lease ownership in the shared database.

Exactly one worker may hold a job's lease.  A lease is only taken over when its
holder has not renewed it for ``lease_seconds``.  Ownership changes increment
the lease ``epoch``, and a worker's identity is the pair
``(worker_id, epoch)``.

The epoch is the fence for checkpoint commits: ``advance_job`` verifies it
inside the same ``BEGIN IMMEDIATE`` transaction that writes the checkpoint.  A
worker whose lease was stolen therefore cannot write anything, even if its
process is alive and still believes it owns the job - it cannot roll the
breakpoint back, and it cannot report a false completion.

A slow batch (one that takes longer than ``lease_seconds``) must not make a
live holder look dead.  While training, a background watchdog thread renews the
lease continuously via :func:`renew_with_heartbeat`, so a batch may take
arbitrarily long without ever expiring the lease.  Renewal is suspended on
purpose between batches (the demo pause hook) and stops entirely when the
process exits, which is what still lets another worker take over after a real
freeze or crash.
"""
from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .db import connect, transaction


@dataclass(frozen=True)
class Lease:
    job_id: str
    worker_id: str
    epoch: int


class LeaseLost(Exception):
    """This worker is no longer allowed to train the job."""


def register_worker(path: Path | str, worker_id: str) -> None:
    now = time.time()
    with transaction(path) as conn:
        conn.execute(
            """
            INSERT INTO workers(worker_id, status, started_at, heartbeat_at)
            VALUES(?, 'starting', ?, ?)
            ON CONFLICT(worker_id) DO UPDATE SET
                status='starting', started_at=excluded.started_at,
                heartbeat_at=excluded.heartbeat_at
            """,
            (worker_id, now, now),
        )


def heartbeat(path: Path | str, worker_id: str, status: str) -> None:
    with transaction(path) as conn:
        conn.execute(
            "UPDATE workers SET heartbeat_at=?, status=? WHERE worker_id=?",
            (time.time(), status, worker_id),
        )


def _allocate_epoch(conn: sqlite3.Connection, job_id: str) -> int:
    """Bump the job's monotonic lease counter and return the new epoch."""
    conn.execute(
        """
        INSERT INTO lease_epochs(job_id, epoch) VALUES(?, 1)
        ON CONFLICT(job_id) DO UPDATE SET epoch=lease_epochs.epoch + 1
        """,
        (job_id,),
    )
    row = conn.execute("SELECT epoch FROM lease_epochs WHERE job_id=?", (job_id,)).fetchone()
    return int(row["epoch"])


def acquire(
    path: Path | str, job_id: str, worker_id: str, lease_seconds: float
) -> Lease | None:
    """Try to own ``job_id``.

    Returns a lease when the job is unowned or its holder's heartbeat is stale.
    If this worker already owns it, the same epoch is renewed and returned, so a
    worker can call this on every poll.  ``None`` means another live worker owns
    the job.  Every new ownership takes a fresh, never-reused epoch.
    """
    now = time.time()
    with transaction(path) as conn:
        row = conn.execute("SELECT * FROM leases WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            epoch = _allocate_epoch(conn, job_id)
            conn.execute(
                "INSERT INTO leases(job_id, worker_id, epoch, heartbeat_at) VALUES(?, ?, ?, ?)",
                (job_id, worker_id, epoch, now),
            )
            return Lease(job_id, worker_id, epoch)
        if row["worker_id"] == worker_id:
            conn.execute("UPDATE leases SET heartbeat_at=? WHERE job_id=?", (now, job_id))
            return Lease(job_id, worker_id, int(row["epoch"]))
        if now - float(row["heartbeat_at"]) < lease_seconds:
            return None
        epoch = _allocate_epoch(conn, job_id)
        conn.execute(
            "UPDATE leases SET worker_id=?, epoch=?, heartbeat_at=? WHERE job_id=?",
            (worker_id, epoch, now, job_id),
        )
        return Lease(job_id, worker_id, epoch)


def renew(path: Path | str, lease: Lease, lease_seconds: float) -> bool:
    """Renew if we still match the lease; ``False`` means it moved on."""
    now = time.time()
    with transaction(path) as conn:
        row = conn.execute(
            "SELECT worker_id, epoch, heartbeat_at FROM leases WHERE job_id=?",
            (lease.job_id,),
        ).fetchone()
        if row is None or row["worker_id"] != lease.worker_id or int(row["epoch"]) != lease.epoch:
            return False
        conn.execute("UPDATE leases SET heartbeat_at=? WHERE job_id=?", (now, lease.job_id))
        return True


def renew_with_heartbeat(
    path: Path | str, lease: Lease, worker_id: str, status: str
) -> bool:
    """Refresh the worker heartbeat and renew the lease in one transaction.

    Used by the worker's background watchdog while a batch is in flight, so a
    batch that takes longer than the lease timeout can never make a live worker
    appear dead.  The renewal is still fenced by ``(worker_id, epoch)``: when
    another instance has taken over, this returns ``False`` and writes nothing.
    """
    now = time.time()
    with transaction(path) as conn:
        row = conn.execute(
            "SELECT worker_id, epoch FROM leases WHERE job_id=?",
            (lease.job_id,),
        ).fetchone()
        if row is None or row["worker_id"] != lease.worker_id or int(row["epoch"]) != lease.epoch:
            return False
        conn.execute(
            "UPDATE workers SET heartbeat_at=?, status=? WHERE worker_id=?",
            (now, status, worker_id),
        )
        conn.execute("UPDATE leases SET heartbeat_at=? WHERE job_id=?", (now, lease.job_id))
        return True


def owner_ok(conn: sqlite3.Connection, lease: Lease) -> bool:
    """Check ownership inside an already-open write transaction."""
    row = conn.execute(
        "SELECT worker_id, epoch FROM leases WHERE job_id=?", (lease.job_id,)
    ).fetchone()
    return bool(row) and row["worker_id"] == lease.worker_id and int(row["epoch"]) == lease.epoch


def release(path: Path | str, lease: Lease) -> None:
    """Give up a lease, but only if it is still ours.

    A stale caller's DELETE matches nothing, so release is always safe.
    """
    with transaction(path) as conn:
        conn.execute(
            "DELETE FROM leases WHERE job_id=? AND worker_id=? AND epoch=?",
            (lease.job_id, lease.worker_id, lease.epoch),
        )


def holder(path: Path | str, job_id: str) -> tuple[str, int, float] | None:
    with connect(path) as conn:
        row = conn.execute(
            "SELECT worker_id, epoch, heartbeat_at FROM leases WHERE job_id=?", (job_id,)
        ).fetchone()
    if row is None:
        return None
    return (row["worker_id"], int(row["epoch"]), float(row["heartbeat_at"]))
