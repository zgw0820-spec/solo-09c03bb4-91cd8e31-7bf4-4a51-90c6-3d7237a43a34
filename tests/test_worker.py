"""In-process worker tests: the same Worker class both containers run."""
from __future__ import annotations

import threading
import time

import numpy as np

from app import lease as lm
from app import store
from app.config import Settings
from app.db import init_db
from app.worker import Worker


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        db_path=tmp_path / "training.db",
        lease_seconds=0.6,
        heartbeat_interval=0.1,
        poll_interval=0.02,
        batch_delay_seconds=0.01,
    )
    base.update(overrides)
    return Settings(**base)


def sample(records: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(records, 3)).tolist()
    labels = [int(i) % 2 for i in range(records)]
    return features, labels


def run_worker(settings: Settings, worker_id: str) -> tuple[Worker, threading.Thread]:
    worker = Worker(settings, worker_id)
    thread = threading.Thread(target=worker.run, daemon=True)
    thread.start()
    return worker, thread


def stop(worker: Worker, thread: threading.Thread) -> None:
    worker.request_stop()
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_worker_trains_job_to_completion(tmp_path):
    settings = make_settings(tmp_path)
    init_db(settings.db_path)
    job = store.create_job(settings.db_path, *sample(20), 2)
    worker, thread = run_worker(settings, "w1")
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            if store.get_job(settings.db_path, job["job_id"])["status"] == "completed":
                break
            time.sleep(0.02)
    finally:
        stop(worker, thread)
    final = store.get_job(settings.db_path, job["job_id"])
    assert final["status"] == "completed"
    assert final["completed_steps"] == final["total_steps"] == 6
    checkpoints = store.list_checkpoints(settings.db_path, job["job_id"])
    assert [c["step"] for c in checkpoints] == [1, 2, 3, 4, 5, 6]
    assert {c["worker_id"] for c in checkpoints} == {"w1"}
    assert all(c["lease_epoch"] == 1 for c in checkpoints)
    assert lm.holder(settings.db_path, job["job_id"]) is None  # released on completion


def test_two_workers_never_train_the_same_job_concurrently(tmp_path):
    settings = make_settings(tmp_path, batch_delay_seconds=0.05)
    init_db(settings.db_path)
    jobs = [store.create_job(settings.db_path, *sample(16, seed=i), 2) for i in range(2)]
    workers = [run_worker(settings, f"w{i}") for i in (1, 2)]
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if all(
                store.get_job(settings.db_path, job["job_id"])["status"] == "completed"
                for job in jobs
            ):
                break
            time.sleep(0.02)
    finally:
        for worker, thread in workers:
            stop(worker, thread)

    for job in jobs:
        final = store.get_job(settings.db_path, job["job_id"])
        assert final["status"] == "completed"
        assert final["completed_steps"] == final["total_steps"]
        checkpoints = store.list_checkpoints(settings.db_path, job["job_id"])
        steps = [c["step"] for c in checkpoints]
        assert steps == list(range(1, final["total_steps"] + 1))
        # Exactly one lease epoch may write a given run of consecutive steps
        # per owner; a takeover would be visible as an epoch increase.
        epochs = [c["lease_epoch"] for c in checkpoints]
        assert epochs == sorted(epochs)


def test_slow_batch_keeps_lease_and_completes(tmp_path):
    """A batch longer than the lease must not trigger a takeover livelock.

    The watchdog renews the lease while the batch is in flight, so a single
    worker keeps epoch 1 for the whole job even though every batch takes more
    than twice the lease timeout.  Before the fix, two workers ping-ponged the
    lease epoch and the job stayed at step 0 forever.
    """
    settings = make_settings(
        tmp_path,
        lease_seconds=0.6,
        heartbeat_interval=0.1,
        batch_delay_seconds=1.5,
    )
    init_db(settings.db_path)
    job = store.create_job(settings.db_path, *sample(24, seed=4), 3)
    workers = [run_worker(settings, f"w{i}") for i in (1, 2)]
    try:
        deadline = time.time() + 30
        while time.time() < deadline:
            if store.get_job(settings.db_path, job["job_id"])["status"] == "completed":
                break
            time.sleep(0.05)
    finally:
        for worker, thread in workers:
            stop(worker, thread)

    final = store.get_job(settings.db_path, job["job_id"])
    assert final["status"] == "completed"
    assert final["completed_steps"] == final["total_steps"] == 9
    checkpoints = store.list_checkpoints(settings.db_path, job["job_id"])
    assert [c["step"] for c in checkpoints] == list(range(1, 10))
    # The live holder kept renewing mid-batch: no takeover ever happened, so
    # the whole job stayed with one owner under one lease epoch.
    assert len({c["worker_id"] for c in checkpoints}) == 1
    assert all(c["lease_epoch"] == 1 for c in checkpoints)
    assert lm.holder(settings.db_path, job["job_id"]) is None


def test_paused_worker_still_loses_lease_for_takeover(tmp_path):
    """Suspending renewal between batches must still let the lease expire."""
    pause_file = tmp_path / "pause"
    settings = make_settings(tmp_path, batch_delay_seconds=0.01)
    init_db(settings.db_path)
    job = store.create_job(settings.db_path, *sample(24, seed=6), 3)

    # Only the first worker honours the pause hook; the takeover worker keeps
    # training normally once it owns the lease.
    first_worker = Worker(settings, "first", pause_file=pause_file)
    first_thread = threading.Thread(target=first_worker.run, daemon=True)
    first_thread.start()
    try:
        # Arm the pause while "first" is still early in training.  It finishes
        # the batch already in flight (its watchdog keeps that lease alive) and
        # then blocks at the inter-batch hook without renewing anything.
        deadline = time.time() + 20
        while time.time() < deadline:
            if store.get_job(settings.db_path, job["job_id"])["completed_steps"] >= 1:
                break
            time.sleep(0.01)
        pause_file.write_text("paused")
        # Wait until "first" is actually inside the pause hook: its watchdog is
        # suspended, so the lease heartbeat goes stale.  From this moment its
        # progress can no longer change.
        deadline = time.time() + 20
        while time.time() < deadline:
            holder = lm.holder(settings.db_path, job["job_id"])
            if holder is not None and time.time() - holder[2] >= settings.lease_seconds:
                break
            time.sleep(0.02)
        frozen = store.get_job(settings.db_path, job["job_id"])
        assert 1 <= frozen["completed_steps"] < frozen["total_steps"]
        second_worker, second_thread = run_worker(settings, "second")
        try:
            # The idle worker must take over the expired lease with a fresh
            # epoch and finish the job.
            deadline = time.time() + 20
            while time.time() < deadline:
                holder = lm.holder(settings.db_path, job["job_id"])
                if holder is not None and holder[0] != "first":
                    break
                time.sleep(0.05)
            deadline = time.time() + 30
            while time.time() < deadline:
                if store.get_job(settings.db_path, job["job_id"])["status"] == "completed":
                    break
                time.sleep(0.02)
        finally:
            stop(second_worker, second_thread)
    finally:
        pause_file.unlink(missing_ok=True)
        stop(first_worker, first_thread)

    final = store.get_job(settings.db_path, job["job_id"])
    assert final["status"] == "completed"
    checkpoints = store.list_checkpoints(settings.db_path, job["job_id"])
    owners = [c["worker_id"] for c in checkpoints]
    assert owners.count("first") == frozen["completed_steps"]
    assert "second" in owners
    epochs = [c["lease_epoch"] for c in checkpoints]
    assert epochs == sorted(epochs) and epochs[-1] > epochs[0]


def test_worker_survives_a_crash_preventing_commit(tmp_path):
    """A worker that loses its lease midway leaves a clean resumable checkpoint."""
    settings = make_settings(tmp_path, batch_delay_seconds=0.0)
    init_db(settings.db_path)
    job = store.create_job(settings.db_path, *sample(24, seed=3), 3)

    lm.register_worker(settings.db_path, "first")
    first = lm.acquire(settings.db_path, job["job_id"], "first", settings.lease_seconds)
    for _ in range(2):
        store.advance_job(settings.db_path, first)
    # "first" dies here; a new worker takes over after expiry.
    time.sleep(settings.lease_seconds + 0.05)

    second_worker, thread = run_worker(settings, "second")
    try:
        deadline = time.time() + 20
        while time.time() < deadline:
            if store.get_job(settings.db_path, job["job_id"])["status"] == "completed":
                break
            time.sleep(0.02)
    finally:
        stop(second_worker, thread)

    final = store.get_job(settings.db_path, job["job_id"])
    assert final["status"] == "completed"
    checkpoints = store.list_checkpoints(settings.db_path, job["job_id"])
    assert [c["step"] for c in checkpoints] == list(range(1, 10))
    owners = [c["worker_id"] for c in checkpoints]
    assert owners[:2] == ["first", "first"]
    assert set(owners[2:]) == {"second"}
    epochs = [c["lease_epoch"] for c in checkpoints]
    assert epochs[:2] == [1, 1]
    assert epochs[2:] == [2] * 7
