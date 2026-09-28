import asyncio
import unittest
from threading import Barrier, Thread

from app.generation_coordinator import (
    GenerationCoordinator,
    GenerationJobCancelledError,
    GenerationStatus,
    SessionGenerationBusyError,
)


class GenerationCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.coordinator = GenerationCoordinator(completed_limit=16)

    async def asyncTearDown(self):
        await self.coordinator.shutdown()

    async def test_fifo_runs_one_job_at_a_time(self):
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        order = []
        active = 0
        peak_active = 0

        async def first_runner(cancel_event, publish):
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            order.append("first-start")
            first_started.set()
            await release_first.wait()
            order.append("first-end")
            active -= 1
            return "first"

        async def second_runner(cancel_event, publish):
            nonlocal active, peak_active
            active += 1
            peak_active = max(peak_active, active)
            order.append("second-start")
            active -= 1
            return "second"

        first, _ = await self.coordinator.submit("session-a", "one", "chat", first_runner)
        await first_started.wait()
        second, _ = await self.coordinator.submit("session-b", "two", "chat", second_runner)
        self.assertEqual(second.status, GenerationStatus.QUEUED)
        self.assertEqual(self.coordinator.status(second)["queue_position"], 1)

        release_first.set()
        self.assertEqual(await first.wait(), "first")
        self.assertEqual(await second.wait(), "second")
        self.assertEqual(order, ["first-start", "first-end", "second-start"])
        self.assertEqual(peak_active, 1)

    async def test_duplicate_request_is_idempotent(self):
        release = asyncio.Event()
        calls = 0

        async def runner(cancel_event, publish):
            nonlocal calls
            calls += 1
            await release.wait()
            return "done"

        first, created = await self.coordinator.submit("session", "request", "chat", runner)
        duplicate, duplicate_created = await self.coordinator.submit(
            "session", "request", "chat", runner
        )
        self.assertTrue(created)
        self.assertFalse(duplicate_created)
        self.assertIs(first, duplicate)
        release.set()
        self.assertEqual(await duplicate.wait(), "done")
        self.assertEqual(calls, 1)

    async def test_one_unfinished_job_per_session(self):
        release = asyncio.Event()

        async def runner(cancel_event, publish):
            await release.wait()

        await self.coordinator.submit("session", "one", "chat", runner)
        with self.assertRaises(SessionGenerationBusyError):
            await self.coordinator.submit("session", "two", "retry", runner)
        release.set()

    async def test_cancel_queued_job_never_runs(self):
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        second_ran = False

        async def first_runner(cancel_event, publish):
            first_started.set()
            await release_first.wait()
            return "first"

        async def second_runner(cancel_event, publish):
            nonlocal second_ran
            second_ran = True

        first, _ = await self.coordinator.submit("session-a", "one", "chat", first_runner)
        await first_started.wait()
        second, _ = await self.coordinator.submit("session-b", "two", "chat", second_runner)

        self.assertEqual(self.coordinator.cancel("session-b", "two"), "cancelled")
        with self.assertRaises(GenerationJobCancelledError):
            await second.wait()
        release_first.set()
        await first.wait()
        await asyncio.sleep(0)
        self.assertFalse(second_ran)

    async def test_cancel_running_job_uses_shared_event(self):
        runner_saw_cancel = asyncio.Event()

        async def runner(cancel_event, publish):
            while not cancel_event.is_set():
                await asyncio.sleep(0)
            runner_saw_cancel.set()
            raise RuntimeError("worker cancellation")

        job, _ = await self.coordinator.submit("session", "request", "chat", runner)
        while job.status == GenerationStatus.QUEUED:
            await asyncio.sleep(0)
        self.assertEqual(self.coordinator.cancel("session", "request"), "cancelling")
        await runner_saw_cancel.wait()
        with self.assertRaises(GenerationJobCancelledError):
            await job.wait()
        self.assertEqual(job.status, GenerationStatus.CANCELLED)

    async def test_events_include_stable_job_identity_and_sequence(self):
        async def runner(cancel_event, publish):
            await publish({"type": "token", "content": "A"})
            await publish({"type": "token", "content": "B"})
            return "AB"

        job, _ = await self.coordinator.submit("session", None, "chat", runner)
        events = [event async for event in job.stream()]
        self.assertEqual(await job.wait(), "AB")
        self.assertEqual(
            [event["type"] for event in events],
            ["started", "token", "token", "completed"],
        )
        self.assertEqual([event["sequence"] for event in events], [1, 2, 3, 4])
        self.assertTrue(all(event["schema_version"] == 1 for event in events))
        self.assertTrue(all(event["job_id"] == job.job_id for event in events))
        self.assertTrue(all(event["request_id"] == job.request_id for event in events))
        self.assertTrue(all(event["session_id"] == "session" for event in events))
        self.assertTrue(all(event["kind"] == "chat" for event in events))
        self.assertTrue(events[-1]["terminal"])
        self.assertEqual(events[-1]["status"], "completed")
        self.assertIsInstance(events[-1]["queue_wait_ms"], int)
        self.assertIsInstance(events[-1]["run_ms"], int)
        self.assertEqual(self.coordinator.status(job)["run_ms"], events[-1]["run_ms"])
        self.assertTrue(job.request_id)

    async def test_global_status_reports_running_and_queue_depth(self):
        started = asyncio.Event()
        release = asyncio.Event()

        async def runner(cancel_event, publish):
            started.set()
            await release.wait()

        first, _ = await self.coordinator.submit("session-a", "one", "chat", runner)
        await started.wait()
        await self.coordinator.submit("session-b", "two", "chat", runner)
        status = self.coordinator.global_status()
        self.assertEqual(status["current_job_id"], first.job_id)
        self.assertEqual(status["queue_depth"], 1)
        release.set()

    async def test_same_request_id_is_scoped_to_session(self):
        async def runner(cancel_event, publish):
            return "done"

        first, first_created = await self.coordinator.submit(
            "session-a", "request", "chat", runner
        )
        second, second_created = await self.coordinator.submit(
            "session-b", "request", "chat", runner
        )

        self.assertTrue(first_created)
        self.assertTrue(second_created)
        self.assertIsNot(first, second)
        self.assertEqual(await first.wait(), "done")
        self.assertEqual(await second.wait(), "done")

    async def test_late_subscriber_replays_published_events(self):
        published = asyncio.Event()
        release = asyncio.Event()

        async def runner(cancel_event, publish):
            await publish({"type": "token", "content": "A"})
            published.set()
            await release.wait()
            await publish({"type": "token", "content": "B"})
            return "AB"

        job, _ = await self.coordinator.submit("session", "request", "chat", runner)
        await published.wait()
        stream = job.stream()
        first = await anext(stream)
        second = await anext(stream)
        self.assertEqual([first["type"], second["type"]], ["started", "token"])
        release.set()
        remaining = [event async for event in stream]
        self.assertEqual(
            [event.get("content") for event in remaining if event["type"] == "token"],
            ["B"],
        )
        self.assertEqual(remaining[-1]["type"], "completed")
        self.assertTrue(remaining[-1]["terminal"])
        self.assertEqual(await job.wait(), "AB")

    async def test_failed_runner_emits_canonical_terminal_error(self):
        async def runner(cancel_event, publish):
            await publish({"type": "token", "content": "partial"})
            raise ValueError("broken")

        job, _ = await self.coordinator.submit("session", "request", "chat", runner)
        events = [event async for event in job.stream()]

        with self.assertRaisesRegex(ValueError, "broken"):
            await job.wait()
        terminal = events[-1]
        self.assertEqual(terminal["type"], "error")
        self.assertEqual(terminal["status"], "failed")
        self.assertEqual(terminal["code"], "GENERATION_FAILED")
        self.assertEqual(terminal["message"], "broken")
        self.assertTrue(terminal["terminal"])
        self.assertEqual([event["sequence"] for event in events], [1, 2, 3])

    async def test_closing_stream_does_not_cancel_generation(self):
        published = asyncio.Event()
        release = asyncio.Event()

        async def runner(cancel_event, publish):
            await publish({"type": "token", "content": "A"})
            published.set()
            await release.wait()
            return "done"

        job, _ = await self.coordinator.submit("session", "request", "chat", runner)
        await published.wait()
        stream = job.stream()
        await anext(stream)  # started
        await anext(stream)  # token
        await stream.aclose()

        self.assertFalse(job.cancel_event.is_set())
        self.assertFalse(job.terminal)
        release.set()
        self.assertEqual(await job.wait(), "done")
        self.assertEqual(job.status, GenerationStatus.COMPLETED)
        self.assertEqual(job.events[-1]["type"], "completed")

    async def test_failed_runner_has_failed_status(self):
        async def runner(cancel_event, publish):
            raise ValueError("broken")

        job, _ = await self.coordinator.submit("session", "request", "chat", runner)
        with self.assertRaisesRegex(ValueError, "broken"):
            await job.wait()
        self.assertEqual(job.status, GenerationStatus.FAILED)

    async def test_completed_retention_is_bounded(self):
        coordinator = GenerationCoordinator(completed_limit=2)

        async def runner(cancel_event, publish):
            return "done"

        jobs = []
        for index in range(3):
            job, _ = await coordinator.submit(
                f"session-{index}", f"request-{index}", "chat", runner
            )
            jobs.append(job)
            await job.wait()

        self.assertIsNone(coordinator.get_job(jobs[0].job_id))
        self.assertIsNotNone(coordinator.get_job(jobs[1].job_id))
        self.assertIsNotNone(coordinator.get_job(jobs[2].job_id))
        await coordinator.shutdown()

    async def test_shutdown_resolves_running_and_queued_jobs(self):
        started = asyncio.Event()

        async def running(cancel_event, publish):
            started.set()
            while not cancel_event.is_set():
                await asyncio.sleep(0)
            raise GenerationJobCancelledError("cancelled")

        async def queued(cancel_event, publish):
            self.fail("queued runner must not start during shutdown")

        first, _ = await self.coordinator.submit("session-a", "one", "chat", running)
        await started.wait()
        second, _ = await self.coordinator.submit("session-b", "two", "chat", queued)

        await self.coordinator.shutdown()

        with self.assertRaises(GenerationJobCancelledError):
            await first.wait()
        with self.assertRaises(GenerationJobCancelledError):
            await second.wait()
        self.assertEqual(first.status, GenerationStatus.CANCELLED)
        self.assertEqual(second.status, GenerationStatus.CANCELLED)
        self.assertEqual(self.coordinator.global_status()["queue_depth"], 0)


class ThreadedAdmissionTests(unittest.TestCase):
    def test_session_transaction_admission_is_atomic(self):
        from app.session_manager import SessionManager

        manager = SessionManager.__new__(SessionManager)
        manager._active_transactions = set()
        manager._transaction_lock = __import__("threading").Lock()
        barrier = Barrier(3)
        results = []

        def enter():
            barrier.wait()
            results.append(manager.begin_transaction("session"))

        threads = [Thread(target=enter), Thread(target=enter)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()

        self.assertEqual(sorted(results), [False, True])


if __name__ == "__main__":
    unittest.main()
