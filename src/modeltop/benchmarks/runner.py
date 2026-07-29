"""Structured bounded-worker scheduling for benchmarks."""

import asyncio
from collections.abc import Awaitable, Callable, Sequence


async def run_bounded_workers[ItemT, ResultT](
    items: Sequence[ItemT],
    concurrency: int,
    worker: Callable[[ItemT], Awaitable[ResultT]],
    on_result: Callable[[ResultT], None],
    should_stop: Callable[[], bool],
) -> None:
    """Process items with a fixed number of lazy, stop-aware workers.

    Items are claimed only when a worker has capacity. A stop request prevents
    subsequent claims without interrupting work that has already started.
    Unexpected worker or result-callback exceptions escape the task group,
    which cancels and awaits every sibling worker.
    """

    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    next_index = 0
    claim_lock = asyncio.Lock()

    async def claim_next_index() -> int | None:
        nonlocal next_index
        async with claim_lock:
            if should_stop() or next_index >= len(items):
                return None
            claimed_index = next_index
            next_index += 1
            return claimed_index

    async def run_worker() -> None:
        while (claimed_index := await claim_next_index()) is not None:
            result = await worker(items[claimed_index])
            on_result(result)

    worker_count = min(concurrency, len(items))
    async with asyncio.TaskGroup() as task_group:
        for _ in range(worker_count):
            task_group.create_task(run_worker())
