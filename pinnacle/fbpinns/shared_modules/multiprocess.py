"""FBPINNs multiprocessing helpers (vendored).

This module provides a small wrapper around :mod:`multiprocessing` used by
the original FBPINNs code. The API is intentionally minimal and keeps the
same behaviour as upstream:

- class :class:`Pool` with a ``starmap`` method,
- the target function receives the worker index ``ip`` as the first argument.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from typing import Any, Callable, Iterable, List, Sequence


class Pool:
    """Multiprocessing pool for running a function across multiple workers.

    This is analogous to :class:`multiprocessing.Pool`, but the target
    function additionally receives the worker index ``ip`` as the first
    argument.
    """

    def __init__(self, processes: int = 1) -> None:
        """Create the pool.

        :param int processes: Number of worker processes.
        """
        self.processes = processes

    def _worker_loop(
        self,
        func: Callable[..., Any],
        ip: int,
        input_queue: "mp.Queue[List[Any]]",
    ) -> None:
        """Worker loop, consuming tasks from ``input_queue``."""
        while True:
            # get task
            args = input_queue.get(block=True, timeout=None)
            if args == -1:
                # poison pill – stop this worker
                break

            # just in case there is a memory leak, run each task
            # in a short-lived process
            p = mp.Process(target=func, args=[ip] + args, daemon=False)
            p.start()
            p.join()
            if p.exitcode != 0:
                print(
                    f"ERROR: process {os.getpid()} "
                    "terminated unexpectedly"
                )
                break

    def starmap(
        self,
        func: Callable[..., Any],
        iterable: Iterable[Sequence[Any]],
    ) -> None:
        """Apply ``func`` to every argument tuple in ``iterable``.

        Behaviour is similar to :meth:`multiprocessing.Pool.starmap`,
        except that the worker index ``ip`` is also passed as the first
        argument, i.e. we compute ``func(ip, *args)``.
        """
        # put all inputs on input queue
        input_queue: "mp.Queue[List[Any]]" = mp.Queue()
        for args in iterable:
            input_queue.put(list(args))
        # poison pills
        for _ in range(self.processes):
            input_queue.put(-1)

        # start processes running
        processes: List[mp.Process] = [
            mp.Process(
                target=self._worker_loop,
                args=(func, ip, input_queue),
                daemon=False,
            )
            for ip in range(self.processes)
        ]
        for p in processes:
            p.start()
        for p in processes:
            p.join()

    def __enter__(self) -> "Pool":
        return self

    def __exit__(self, *args: Any) -> None:
        # nothing special to clean up: workers are short-lived
        # and joined in :meth:`starmap`.
        return None


if __name__ == "__main__":  # pragma: no cover - simple usage example
    import numpy as np

    def _example(ip: int, x: int, y: int) -> None:
        print(f"[worker {ip}] {x} * {y} = {x * y}")

    with Pool(processes=4) as pool:
        pool.starmap(_example, zip(np.arange(4), np.arange(4)))
