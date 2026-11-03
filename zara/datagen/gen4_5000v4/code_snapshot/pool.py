"""Process pool with a watchdog for FreeGSNKE workers (the solver hangs in C code ~1 in 150 solves; SIGALRM cannot interrupt).

Each worker k calls beat(dir, k, tag) before every item; the parent kills a worker whose heartbeat is older than `stall`
seconds, calls on_hang(k, tag) (e.g. write a fallback output for that item so the restarted worker skips it), and restarts it.
"""
from __future__ import annotations

import os
import time
from multiprocessing import Process


def beat(hb_dir, k, tag):
    with open(os.path.join(hb_dir, f"hb.w{k}"), "w") as f:
        f.write(str(tag))


def _read(hb_dir, k):
    p = os.path.join(hb_dir, f"hb.w{k}")
    try:
        return os.path.getmtime(p), open(p).read().strip()
    except OSError:
        return None, None


def run_pool(target, n_workers, args, hb_dir, stall=90.0, on_hang=None, max_restarts=50, poll=5.0):
    """target(k, n_workers, *args) runs worker k. Returns the number of hang restarts."""
    os.makedirs(hb_dir, exist_ok=True)
    ps = {}
    t_start = {}
    restarts = {k: 0 for k in range(n_workers)}
    for k in range(n_workers):
        ps[k] = Process(target=target, args=(k, n_workers) + tuple(args)); ps[k].start(); t_start[k] = time.time()
    n_hang = 0
    while ps:
        time.sleep(poll)
        for k in list(ps):
            p = ps[k]
            if not p.is_alive():
                p.join(); del ps[k]
                continue
            mt, tag = _read(hb_dir, k)
            ref = max(mt or 0, t_start[k])
            if time.time() - ref > stall:
                p.kill(); p.join()
                n_hang += 1
                print(f"[pool] worker {k} stalled on {tag}; restart {restarts[k] + 1}", flush=True)
                if on_hang is not None and tag:
                    try:
                        on_hang(k, tag)
                    except Exception as ex:  # noqa: BLE001
                        print(f"[pool] on_hang failed: {ex}", flush=True)
                restarts[k] += 1
                if restarts[k] > max_restarts:
                    del ps[k]
                    continue
                ps[k] = Process(target=target, args=(k, n_workers) + tuple(args)); ps[k].start(); t_start[k] = time.time()
    return n_hang
