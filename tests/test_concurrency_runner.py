"""Controlled-event contracts for the bounded benchmark worker pool."""

import asyncio

import pytest

from modeltop.benchmarks.runner import run_bounded_workers


def test_pool_bounds_tasks_reuses_slots_and_reports_completion_order() -> None:
    async def scenario() -> None:
        item_count = 12
        releases = [asyncio.Event() for _ in range(item_count)]
        started = [asyncio.Event() for _ in range(item_count)]
        reported = [asyncio.Event() for _ in range(item_count)]
        calls: list[int] = []
        completion_order: list[int] = []
        active = 0
        max_active = 0

        async def worker(item: int) -> int:
            nonlocal active, max_active
            calls.append(item)
            active += 1
            max_active = max(max_active, active)
            started[item].set()
            try:
                await releases[item].wait()
                return item
            finally:
                active -= 1

        def on_result(item: int) -> None:
            completion_order.append(item)
            reported[item].set()

        runner_task = asyncio.create_task(
            run_bounded_workers(
                tuple(range(item_count)),
                4,
                worker,
                on_result,
                lambda: False,
            )
        )
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in started[:4])), timeout=1
        )

        current_task = asyncio.current_task()
        assert current_task is not None
        worker_tasks = asyncio.all_tasks() - {current_task, runner_task}
        assert len(worker_tasks) == 4
        assert set(calls) == {0, 1, 2, 3}
        assert active == 4

        releases[2].set()
        await asyncio.wait_for(started[4].wait(), timeout=1)
        assert completion_order == [2]
        assert active == 4

        release_order = [0, 4, 1, 3, 5, 6, 7, 8, 9, 10, 11]
        for item in release_order:
            await asyncio.wait_for(started[item].wait(), timeout=1)
            releases[item].set()
            await asyncio.wait_for(reported[item].wait(), timeout=1)

        await asyncio.wait_for(runner_task, timeout=1)

        assert completion_order == [2, *release_order]
        assert sorted(calls) == list(range(item_count))
        assert len(calls) == item_count
        assert max_active == 4
        assert active == 0

    asyncio.run(scenario())


def test_stop_prevents_workers_from_claiming_more_items() -> None:
    async def scenario() -> None:
        releases = [asyncio.Event() for _ in range(8)]
        started = [asyncio.Event() for _ in range(8)]
        stop_requested = False
        calls: list[int] = []
        results: list[int] = []

        async def worker(item: int) -> int:
            calls.append(item)
            started[item].set()
            await releases[item].wait()
            return item

        runner_task = asyncio.create_task(
            run_bounded_workers(
                tuple(range(8)),
                3,
                worker,
                results.append,
                lambda: stop_requested,
            )
        )
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in started[:3])), timeout=1
        )

        stop_requested = True
        for release in releases[:3]:
            release.set()
        await asyncio.wait_for(runner_task, timeout=1)

        assert set(calls) == {0, 1, 2}
        assert sorted(results) == [0, 1, 2]
        assert not any(event.is_set() for event in started[3:])

    asyncio.run(scenario())


def test_cancelling_pool_cleans_up_workers_without_further_claims() -> None:
    async def scenario() -> None:
        started = [asyncio.Event() for _ in range(8)]
        baseline_tasks = asyncio.all_tasks()
        active: set[int] = set()
        cancelled: set[int] = set()

        async def worker(item: int) -> int:
            active.add(item)
            started[item].set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.add(item)
                raise
            finally:
                active.remove(item)
            raise AssertionError("worker unexpectedly returned")

        runner_task = asyncio.create_task(
            run_bounded_workers(
                tuple(range(8)),
                3,
                worker,
                lambda result: None,
                lambda: False,
            )
        )
        await asyncio.wait_for(
            asyncio.gather(*(event.wait() for event in started[:3])), timeout=1
        )

        runner_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await runner_task

        assert cancelled == {0, 1, 2}
        assert not active
        assert not any(event.is_set() for event in started[3:])
        assert asyncio.all_tasks() == baseline_tasks

    asyncio.run(scenario())


def test_unexpected_worker_exception_cancels_siblings_without_leaks() -> None:
    async def scenario() -> None:
        all_started = asyncio.Event()
        baseline_tasks = asyncio.all_tasks()
        started: set[int] = set()
        cancelled: set[int] = set()
        exited: set[int] = set()

        async def worker(item: int) -> int:
            started.add(item)
            if len(started) == 3:
                all_started.set()
            try:
                await all_started.wait()
                if item == 0:
                    raise RuntimeError("worker failed")
                await asyncio.Event().wait()
                return item
            except asyncio.CancelledError:
                cancelled.add(item)
                raise
            finally:
                exited.add(item)

        with pytest.raises(ExceptionGroup) as caught:
            await run_bounded_workers(
                tuple(range(5)),
                3,
                worker,
                lambda result: None,
                lambda: False,
            )

        assert len(caught.value.exceptions) == 1
        assert isinstance(caught.value.exceptions[0], RuntimeError)
        assert str(caught.value.exceptions[0]) == "worker failed"
        assert started == {0, 1, 2}
        assert cancelled == {1, 2}
        assert exited == {0, 1, 2}
        assert asyncio.all_tasks() == baseline_tasks

    asyncio.run(scenario())
