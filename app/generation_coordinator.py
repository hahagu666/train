"""Single-model FIFO generation scheduling and cancellation."""
from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from threading import Event
from time import monotonic
from typing import Any, Awaitable, Callable, Dict, Optional
from uuid import uuid4


SCHEMA_VERSION = 1
_TERMINAL_EVENT = object()
_SHUTDOWN_JOB = object()


class GenerationStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    CANCELLING = "cancelling"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    FAILED = "failed"


class SessionGenerationBusyError(RuntimeError):
    pass


class GenerationJobCancelledError(RuntimeError):
    pass


EventPublisher = Callable[[Dict[str, Any]], Awaitable[None]]
GenerationRunner = Callable[[Event, EventPublisher], Awaitable[Any]]


@dataclass
class GenerationJob:
    job_id: str
    request_id: str
    session_id: str
    kind: str
    runner: GenerationRunner = field(repr=False)
    status: GenerationStatus = GenerationStatus.QUEUED
    cancel_event: Event = field(default_factory=Event, repr=False)
    created_at: float = field(default_factory=monotonic)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    result: Any = field(default=None, repr=False)
    error: Optional[BaseException] = field(default=None, repr=False)
    events: list[Dict[str, Any]] = field(default_factory=list, repr=False)
    _future: Optional[asyncio.Future] = field(default=None, repr=False)
    _subscribers: list[asyncio.Queue] = field(default_factory=list, repr=False)
    _sequence: int = 0

    @property
    def terminal(self) -> bool:
        return self.status in {
            GenerationStatus.CANCELLED,
            GenerationStatus.COMPLETED,
            GenerationStatus.FAILED,
        }

    @property
    def queue_wait_ms(self) -> Optional[int]:
        if self.started_at is None:
            return None
        return max(0, round((self.started_at - self.created_at) * 1000))

    @property
    def run_ms(self) -> Optional[int]:
        if self.started_at is None:
            return None
        end = self.finished_at if self.finished_at is not None else monotonic()
        return max(0, round((end - self.started_at) * 1000))

    def snapshot(self, queue_position: Optional[int] = None) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "job_id": self.job_id,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "kind": self.kind,
            "status": self.status.value,
            "queue_position": queue_position,
            "queue_wait_ms": self.queue_wait_ms,
            "run_ms": self.run_ms,
        }

    async def wait(self) -> Any:
        if self._future is None:
            raise RuntimeError("生成任务尚未提交")
        await asyncio.shield(self._future)
        if self.status == GenerationStatus.COMPLETED:
            return self.result
        if self.error is not None:
            try:
                error = type(self.error)(*self.error.args)
            except Exception:
                error = RuntimeError(str(self.error))
            raise error
        raise RuntimeError("生成任务未正常完成")

    async def stream(self):
        queue: asyncio.Queue = asyncio.Queue()
        for event in self.events:
            queue.put_nowait(event)
        if self.terminal:
            queue.put_nowait(_TERMINAL_EVENT)
        else:
            self._subscribers.append(queue)
        try:
            while True:
                event = await queue.get()
                if event is _TERMINAL_EVENT:
                    return
                yield event
        finally:
            if queue in self._subscribers:
                self._subscribers.remove(queue)


class GenerationCoordinator:
    """Run one generation at a time while preserving FIFO request order."""

    def __init__(self, completed_limit: int = 256):
        self._queue: asyncio.Queue[GenerationJob] = asyncio.Queue()
        self._jobs: Dict[str, GenerationJob] = {}
        self._requests: OrderedDict[tuple[str, str], GenerationJob] = OrderedDict()
        self._session_jobs: Dict[str, GenerationJob] = {}
        self._worker_task: Optional[asyncio.Task] = None
        self._completed_limit = max(1, completed_limit)

    async def submit(
        self,
        session_id: str,
        request_id: Optional[str],
        kind: str,
        runner: GenerationRunner,
    ) -> tuple[GenerationJob, bool]:
        self._ensure_worker()
        effective_request_id = request_id or uuid4().hex
        request_key = (session_id, effective_request_id)
        existing = self._requests.get(request_key)
        if existing is not None:
            return existing, False

        active = self._session_jobs.get(session_id)
        if active is not None and not active.terminal:
            raise SessionGenerationBusyError("该会话已有未完成的生成任务")

        loop = asyncio.get_running_loop()
        job = GenerationJob(
            job_id=uuid4().hex,
            request_id=effective_request_id,
            session_id=session_id,
            kind=kind,
            runner=runner,
            _future=loop.create_future(),
        )
        self._jobs[job.job_id] = job
        self._requests[request_key] = job
        self._session_jobs[session_id] = job
        await self._queue.put(job)
        return job, True

    def cancel(self, session_id: str, request_id: str) -> str:
        job = self._requests.get((session_id, request_id))
        if job is None or job.terminal:
            return "finished"
        job.cancel_event.set()
        if job.status == GenerationStatus.QUEUED:
            job.status = GenerationStatus.CANCELLED
            job.finished_at = monotonic()
            job.error = GenerationJobCancelledError("生成已取消")
            self._publish_now(job, self._terminal_event(job))
            self._resolve_job(job)
            self._finish_streams(job)
            self._release_session(job)
            self._trim_completed()
            return "cancelled"
        job.status = GenerationStatus.CANCELLING
        return "cancelling"

    def get_job(self, job_id: str) -> Optional[GenerationJob]:
        return self._jobs.get(job_id)

    def get_request(self, session_id: str, request_id: str) -> Optional[GenerationJob]:
        return self._requests.get((session_id, request_id))

    def get_session_job(self, session_id: str) -> Optional[GenerationJob]:
        return self._session_jobs.get(session_id)

    def status(self, job: GenerationJob) -> Dict[str, Any]:
        return job.snapshot(self._queue_position(job))

    def global_status(self) -> Dict[str, Any]:
        running = next(
            (job for job in self._session_jobs.values() if job.status in {
                GenerationStatus.RUNNING, GenerationStatus.CANCELLING,
            }),
            None,
        )
        queued = sum(
            1 for job in self._session_jobs.values()
            if job.status == GenerationStatus.QUEUED
        )
        return {
            "queue_depth": queued,
            "current_job_id": running.job_id if running else None,
            "current_session_id": running.session_id if running else None,
        }

    async def shutdown(self) -> None:
        active_jobs = list(self._session_jobs.values())
        for job in active_jobs:
            if not job.terminal:
                self.cancel(job.session_id, job.request_id)

        task = self._worker_task
        if task and not task.done():
            # Wake an idle worker without cancelling it while it is unwinding a
            # runner exception. This avoids a Python 3.10 coroutine-finalization
            # edge case and still lets shutdown wait for deterministic cleanup.
            await self._queue.put(_SHUTDOWN_JOB)
            await task
        self._worker_task = None

        # Cancelled queued jobs remain as placeholders until the worker consumes
        # them. Drain those placeholders so queue.join() cannot hang at shutdown.
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self._queue.task_done()
        self._trim_completed()

    def _ensure_worker(self) -> None:
        if self._worker_task is None or self._worker_task.done():
            self._worker_task = asyncio.create_task(self._worker())

    async def _worker(self) -> None:
        while True:
            job = await self._queue.get()
            if job is _SHUTDOWN_JOB:
                self._queue.task_done()
                return
            try:
                if job.status == GenerationStatus.CANCELLED:
                    continue
                job.status = GenerationStatus.RUNNING
                job.started_at = monotonic()
                await self._publish(job, {"type": "started"})
                if job.cancel_event.is_set():
                    raise GenerationJobCancelledError("生成已取消")
                job.result = await job.runner(
                    job.cancel_event,
                    lambda event: self._publish(job, event),
                )
                if job.cancel_event.is_set():
                    raise GenerationJobCancelledError("生成已取消")
                job.status = GenerationStatus.COMPLETED
            except (GenerationJobCancelledError, asyncio.CancelledError) as exc:
                job.status = GenerationStatus.CANCELLED
                job.error = GenerationJobCancelledError("生成已取消")
                if isinstance(exc, asyncio.CancelledError):
                    raise
            except Exception as exc:
                if job.cancel_event.is_set():
                    job.status = GenerationStatus.CANCELLED
                    job.error = GenerationJobCancelledError("生成已取消")
                else:
                    job.status = GenerationStatus.FAILED
                    job.error = exc
            finally:
                job.finished_at = monotonic()
                await self._publish(job, self._terminal_event(job))
                self._resolve_job(job)
                self._finish_streams(job)
                self._release_session(job)
                self._trim_completed()
                self._queue.task_done()

    @staticmethod
    def _terminal_event(job: GenerationJob) -> Dict[str, Any]:
        timing = {
            "queue_wait_ms": job.queue_wait_ms,
            "run_ms": job.run_ms,
        }
        if job.status == GenerationStatus.COMPLETED:
            event = {
                "type": "completed",
                "status": job.status.value,
                "terminal": True,
                **timing,
            }
            if isinstance(job.result, dict):
                event.update(job.result)
            else:
                event["result"] = job.result
            return event
        if job.status == GenerationStatus.CANCELLED:
            return {
                "type": "cancelled",
                "status": job.status.value,
                "terminal": True,
                "code": "CANCELLED",
                "message": str(job.error or "生成已取消"),
                **timing,
            }
        error = job.error
        return {
            "type": "error",
            "status": job.status.value,
            "terminal": True,
            "code": getattr(error, "code", "GENERATION_FAILED"),
            "message": str(error or "生成失败"),
            **timing,
        }

    async def _publish(self, job: GenerationJob, event: Dict[str, Any]) -> None:
        self._publish_now(job, event)

    @staticmethod
    def _publish_now(job: GenerationJob, event: Dict[str, Any]) -> None:
        job._sequence += 1
        enriched = dict(event)
        enriched.update({
            "schema_version": SCHEMA_VERSION,
            "sequence": job._sequence,
            "job_id": job.job_id,
            "request_id": job.request_id,
            "session_id": job.session_id,
            "kind": job.kind,
        })
        job.events.append(enriched)
        for queue in list(job._subscribers):
            queue.put_nowait(enriched)

    @staticmethod
    def _finish_streams(job: GenerationJob) -> None:
        for queue in list(job._subscribers):
            queue.put_nowait(_TERMINAL_EVENT)

    @staticmethod
    def _resolve_job(job: GenerationJob) -> None:
        future = job._future
        if future is None or future.done():
            return
        future.set_result(None)

    def _release_session(self, job: GenerationJob) -> None:
        if self._session_jobs.get(job.session_id) is job:
            del self._session_jobs[job.session_id]

    def _queue_position(self, target: GenerationJob) -> Optional[int]:
        if target.status != GenerationStatus.QUEUED:
            return None
        position = 0
        for job in list(self._queue._queue):
            if job.status != GenerationStatus.QUEUED:
                continue
            position += 1
            if job is target:
                return position
        return None

    def _trim_completed(self) -> None:
        terminal_keys = [
            key for key, job in self._requests.items() if job.terminal
        ]
        while len(terminal_keys) > self._completed_limit:
            key = terminal_keys.pop(0)
            job = self._requests.pop(key, None)
            if job is not None:
                self._jobs.pop(job.job_id, None)
