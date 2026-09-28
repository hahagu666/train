import asyncio
import multiprocessing as mp
import queue
import time
import unittest
from threading import Event, Thread
from unittest.mock import patch

import llm_qwen


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def __getitem__(self, key):
        if isinstance(key, tuple):
            return self
        if isinstance(key, slice):
            return FakeTensor(self.values[key])
        return self.values[key]


class FakeInputs(dict):
    def to(self, device):
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "prompt"

    def __call__(self, text, return_tensors="pt"):
        return FakeInputs(input_ids=FakeTensor([1, 2]))

    def decode(self, generated_ids, skip_special_tokens=True):
        return "reply"


class FakeModel:
    device = "cuda"

    def __init__(self, on_generate=None):
        self.on_generate = on_generate
        self.last_kwargs = None

    def generate(self, **kwargs):
        self.last_kwargs = kwargs
        if self.on_generate:
            self.on_generate(kwargs["stopping_criteria"])
        return [FakeTensor([1, 2, 3])]


class FakeStreamer:
    def __init__(self, *args, **kwargs):
        self.items = queue.Queue()

    def on_finalized_text(self, text, stream_end=False):
        if text:
            self.items.put(text)
        if stream_end:
            self.items.put(StopIteration)

    def __next__(self):
        item = self.items.get(timeout=1.0)
        if item is StopIteration:
            raise StopIteration
        return item


class FakeStreamingModel:
    device = "cuda"

    def __init__(self, generated=None):
        self.generated = generated or [1, 2, 3, 1]

    def generate(self, **kwargs):
        kwargs["streamer"].on_finalized_text("完整回复", stream_end=True)
        return [FakeTensor(self.generated)]


def fake_supervisor_worker(command_queue, result_queue, cancel_signal, device=None):
    """Spawned fake: first worker blocks forever, replacements answer."""
    process_number = int(device or 1)
    result_queue.put(("ready", True, "", {"device": f"fake-{process_number}"}))
    while True:
        command = command_queue.get()
        if command[0] == "shutdown":
            return
        command_kind, request_id, args = command
        prompt = args[0]
        if prompt == "block":
            while True:
                time.sleep(1.0)
        if command_kind == "generate_stream":
            result_queue.put(("stream_chunk", request_id, "A"))
            if prompt == "stream-block":
                while not cancel_signal.is_set():
                    time.sleep(0.01)
                result_queue.put(("result", request_id, "cancelled", "cancelled", {
                    "stop_reason": "cancelled",
                }))
                continue
            if prompt == "stream-length":
                result_queue.put(("result", request_id, "length", "length limit", {
                    "stop_reason": "length",
                }))
                continue
            time.sleep(0.05)
            result_queue.put(("stream_chunk", request_id, "B"))
            result_queue.put(("result", request_id, "ok", "", {
                "stop_reason": "eos",
            }))
            continue
        result_queue.put(("result", request_id, "ok", f"reply:{prompt}", {
            "stop_reason": "eos",
        }))


class CountingSpawnContext:
    def __init__(self):
        self.real = mp.get_context("spawn")
        self.started = []

    def Queue(self):
        return self.real.Queue()

    def Event(self):
        return self.real.Event()

    def Process(self, target, args, name):
        numbered_args = (*args[:-1], len(self.started) + 1)
        process = self.real.Process(target=target, args=numbered_args, name=name)
        self.started.append(process)
        return process


class FakeUnkillableProcess:
    exitcode = None

    def __init__(self):
        self.terminate_calls = 0
        self.kill_calls = 0

    def is_alive(self):
        return True

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1

    def join(self, timeout=None):
        return None


class FakeTimedOutStartupProcess:
    exitcode = None

    def __init__(self):
        self.alive = True

    def start(self):
        return None

    def is_alive(self):
        return self.alive

    def terminate(self):
        self.alive = False

    def kill(self):
        self.alive = False

    def join(self, timeout=None):
        return None


class TimedOutStartupContext:
    def __init__(self):
        self.process = FakeTimedOutStartupProcess()

    def Queue(self):
        result = unittest.mock.MagicMock()
        result.get.side_effect = queue.Empty
        return result

    def Event(self):
        return unittest.mock.MagicMock()

    def Process(self, target, args, name):
        return self.process


class GenerationTests(unittest.TestCase):
    def generate_with(self, model, cancel_event=None, timeout=120.0):
        with patch.object(llm_qwen, "_model_loaded", True), patch.object(
            llm_qwen, "_model", model
        ), patch.object(llm_qwen, "_tokenizer", FakeTokenizer()), patch.object(
            llm_qwen.torch, "no_grad", return_value=unittest.mock.MagicMock(
                __enter__=lambda self: None, __exit__=lambda self, *args: None
            )
        ):
            return llm_qwen._generate_unlocked(
                "hello", cancel_event=cancel_event, timeout=timeout
            )

    def test_complete_generation_returns_decoded_response(self):
        self.assertEqual(self.generate_with(FakeModel()), "reply")

    def test_normal_stream_completion_does_not_set_external_cancel_event(self):
        cancel = Event()

        async def scenario():
            with patch.object(llm_qwen, "_model_loaded", True), patch.object(
                llm_qwen, "_model", FakeStreamingModel()
            ), patch.object(llm_qwen, "_tokenizer", FakeTokenizer()), patch.object(
                llm_qwen, "TextIteratorStreamer", FakeStreamer
            ):
                return [chunk async for chunk in llm_qwen._generate_stream_unlocked(
                    "hello", cancel_event=cancel, timeout=1.0
                )]

        self.assertEqual(asyncio.run(scenario()), ["完整回复"])
        self.assertFalse(cancel.is_set())
        self.assertEqual(llm_qwen._last_generation["output_tokens"], 2)
        self.assertEqual(llm_qwen._last_generation["stop_reason"], "eos")

    def test_stream_length_limit_raises_after_partial_text(self):
        chunks = []

        async def scenario():
            with patch.object(llm_qwen, "_model_loaded", True), patch.object(
                llm_qwen, "_model", FakeStreamingModel([1, 2, 8, 9])
            ), patch.object(llm_qwen, "_tokenizer", FakeTokenizer()), patch.object(
                llm_qwen, "TextIteratorStreamer", FakeStreamer
            ):
                async for chunk in llm_qwen._generate_stream_unlocked(
                    "hello", max_new_tokens=2, timeout=1.0
                ):
                    chunks.append(chunk)

        with self.assertRaises(llm_qwen.GenerationLengthLimitError):
            asyncio.run(scenario())
        self.assertEqual(chunks, ["完整回复"])
        self.assertEqual(llm_qwen._last_generation["output_tokens"], 2)
        self.assertEqual(llm_qwen._last_generation["stop_reason"], "length")

    def test_complete_generation_length_limit_is_not_success(self):
        with self.assertRaises(llm_qwen.GenerationLengthLimitError):
            with patch.object(llm_qwen, "_model_loaded", True), patch.object(
                llm_qwen, "_model", FakeModel()
            ), patch.object(llm_qwen, "_tokenizer", FakeTokenizer()), patch.object(
                llm_qwen.torch, "no_grad", return_value=unittest.mock.MagicMock(
                    __enter__=lambda self: None, __exit__=lambda self, *args: None
                )
            ):
                llm_qwen._generate_unlocked("hello", max_new_tokens=1, timeout=1.0)
        self.assertEqual(llm_qwen._last_generation["stop_reason"], "length")

    def test_cancel_criteria_raises_cancelled_error(self):
        cancel = Event()

        def cancel_during_generate(criteria):
            cancel.set()
            criteria[0](None, None)

        with self.assertRaises(llm_qwen.GenerationCancelledError):
            self.generate_with(FakeModel(cancel_during_generate), cancel_event=cancel)

    def test_deadline_raises_timeout_error(self):
        def reach_deadline(criteria):
            criteria[0].deadline = time.monotonic() - 1.0
            criteria[0](None, None)

        with self.assertRaises(llm_qwen.GenerationTimeoutError):
            self.generate_with(FakeModel(reach_deadline), timeout=1.0)

    def test_deadline_records_timeout_telemetry(self):
        def reach_deadline(criteria):
            criteria[0].deadline = time.monotonic() - 1.0
            criteria[0](None, None)

        with self.assertRaises(llm_qwen.GenerationTimeoutError):
            self.generate_with(FakeModel(reach_deadline), timeout=1.0)
        self.assertEqual(llm_qwen._last_generation["prompt_tokens"], 2)
        self.assertEqual(llm_qwen._last_generation["output_tokens"], 0)
        self.assertEqual(llm_qwen._last_generation["stop_reason"], "timeout")

    def test_warmup_runs_one_deterministic_token(self):
        model = FakeModel()
        with patch.object(llm_qwen, "_model_loaded", True), patch.object(
            llm_qwen, "_device", "cuda"
        ), patch.object(llm_qwen, "_model", model), patch.object(
            llm_qwen, "_tokenizer", FakeTokenizer()
        ), patch.object(llm_qwen.torch, "no_grad", return_value=unittest.mock.MagicMock(
            __enter__=lambda self: None, __exit__=lambda self, *args: None
        )):
            llm_qwen._warmup_model()
        self.assertEqual(model.last_kwargs["max_new_tokens"], 1)
        self.assertFalse(model.last_kwargs["do_sample"])
        self.assertTrue(model.last_kwargs["use_cache"])

    def test_generate_releases_slot_after_error(self):
        with patch.object(llm_qwen, "_IN_MODEL_WORKER", True), patch.object(
            llm_qwen, "_acquire_generation"
        ), patch.object(
            llm_qwen, "_generate_unlocked", side_effect=RuntimeError("failed")
        ), patch.object(llm_qwen._generation_slot, "release") as release:
            with self.assertRaises(RuntimeError):
                llm_qwen.generate("hello")
            release.assert_called_once_with()

    def test_auto_mode_uses_4bit_on_8gb_gpu(self):
        with patch.object(llm_qwen.torch.cuda, "is_available", return_value=True), patch.object(
            llm_qwen, "_get_available_vram_gb", return_value=8.0
        ), patch.object(llm_qwen, "MAIN_MODEL_DTYPE", "auto"):
            self.assertEqual(llm_qwen._select_load_mode(), ("cuda", "4bit"))

    def test_auto_mode_allows_fp16_with_safe_free_vram(self):
        with patch.object(llm_qwen.torch.cuda, "is_available", return_value=True), patch.object(
            llm_qwen, "_get_available_vram_gb", return_value=20.0
        ), patch.object(llm_qwen, "MAIN_MODEL_DTYPE", "auto"):
            self.assertEqual(llm_qwen._select_load_mode(), ("cuda", "fp16"))

    def test_generation_enables_kv_cache(self):
        model = FakeModel()
        self.generate_with(model)
        self.assertTrue(model.last_kwargs["use_cache"])


class SupervisorTests(unittest.TestCase):
    def test_stream_cold_start_uses_request_deadline(self):
        context = TimedOutStartupContext()
        supervisor = llm_qwen._ModelWorkerSupervisor(
            context=context, worker_target=fake_supervisor_worker,
            start_timeout=30.0,
        )

        async def scenario():
            async for _ in supervisor.generate_stream(
                "stream", "", 1, 0.0, None, 0.05,
            ):
                pass

        started = time.monotonic()
        with self.assertRaises(llm_qwen.GenerationTimeoutError):
            asyncio.run(scenario())
        self.assertLess(time.monotonic() - started, 1.0)
        self.assertFalse(context.process.is_alive())

    def test_stream_timeout_does_not_wait_for_replacement_startup(self):
        supervisor = llm_qwen._ModelWorkerSupervisor()
        supervisor._process = FakeUnkillableProcess()
        supervisor._commands = unittest.mock.MagicMock()
        supervisor._results = unittest.mock.MagicMock()
        supervisor._results.get.side_effect = queue.Empty
        supervisor._cancel_signal = unittest.mock.MagicMock()
        supervisor._state = "ready"

        replacement_started = Event()

        def schedule_recovery():
            supervisor._process = None
            supervisor._state = "recovering"
            replacement_started.set()

        async def scenario():
            async for _ in supervisor.generate_stream(
                "stream", "", 1, 0.0, None, 0.05,
            ):
                pass

        with patch.object(supervisor, "_start_locked", return_value=True), patch.object(
            supervisor, "_schedule_recovery_after_timeout_locked",
            side_effect=schedule_recovery,
        ):
            with self.assertRaises(llm_qwen.GenerationTimeoutError):
                asyncio.run(scenario())
        self.assertTrue(replacement_started.is_set())
        self.assertEqual(supervisor.status()["state"], "recovering")

    def test_hard_timeout_reaps_old_worker_and_readies_replacement(self):
        context = CountingSpawnContext()
        supervisor = llm_qwen._ModelWorkerSupervisor(
            context=context, worker_target=fake_supervisor_worker, start_timeout=30.0,
        )
        completed = Event()
        observed = {}

        def run_blocked():
            try:
                supervisor.generate("block", "", 1, 0.0, None, 0.15)
            except BaseException as exc:
                observed["error"] = exc
                observed["old_dead_at_return"] = not context.started[0].is_alive()
                observed["replacement_ready_at_return"] = supervisor.status()["state"] == "ready"
            finally:
                completed.set()

        try:
            thread = Thread(target=run_blocked)
            thread.start()
            deadline = time.monotonic() + 30.0
            while supervisor.status()["state"] != "generating" and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(supervisor.status()["state"], "generating")
            self.assertTrue(context.started[0].is_alive())
            self.assertFalse(completed.is_set(), "timeout resolved while blocked worker was alive")
            thread.join(30.0)
            self.assertFalse(thread.is_alive())
            self.assertIsInstance(observed.get("error"), llm_qwen.GenerationTimeoutError)
            self.assertTrue(observed["old_dead_at_return"])
            self.assertTrue(observed["replacement_ready_at_return"])
            self.assertEqual(len(context.started), 2)
            self.assertFalse(context.started[0].is_alive())
            self.assertTrue(context.started[1].is_alive())
            self.assertEqual(supervisor.generate("next", "", 1, 0.0, None, 1.0), "reply:next")
        finally:
            supervisor.unload()

    def test_timeout_keeps_ownership_when_worker_cannot_be_reaped(self):
        supervisor = llm_qwen._ModelWorkerSupervisor()
        process = FakeUnkillableProcess()
        supervisor._process = process
        supervisor._commands = unittest.mock.MagicMock()
        supervisor._results = unittest.mock.MagicMock()
        supervisor._results.get.side_effect = queue.Empty
        supervisor._cancel_signal = unittest.mock.MagicMock()
        supervisor._state = "generating"

        with patch.object(supervisor, "_start_locked", return_value=True):
            with self.assertRaises(llm_qwen.ModelWorkerUnterminableError):
                supervisor.generate("block", "", 1, 0.0, None, 0.001)

        status = supervisor.status()
        self.assertEqual(status["state"], "fatal_worker_unterminable")
        self.assertTrue(status["busy"])
        self.assertTrue(process.is_alive())
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 1)
        with self.assertRaises(llm_qwen.ModelBusyError):
            supervisor.generate("next", "", 1, 0.0, None, 1.0)

    def test_timeout_reports_recovery_failure_after_old_worker_dies(self):
        supervisor = llm_qwen._ModelWorkerSupervisor()
        supervisor._state = "generating"
        supervisor._process = FakeUnkillableProcess()
        supervisor._results = unittest.mock.MagicMock()
        supervisor._results.get.side_effect = queue.Empty
        supervisor._cancel_signal = unittest.mock.MagicMock()
        calls = []

        def fail_recovery():
            calls.append("recover")
            supervisor._state = "recovery_failed"
            raise llm_qwen.ModelWorkerRecoveryError("replacement failed")

        with patch.object(supervisor, "_start_locked", return_value=True), patch.object(
            supervisor, "_commands", unittest.mock.MagicMock()
        ), patch.object(supervisor, "_replace_after_timeout_locked", side_effect=fail_recovery):
            with self.assertRaises(llm_qwen.ModelWorkerRecoveryError):
                supervisor.generate("block", "", 1, 0.0, None, 0.001)

        self.assertEqual(calls, ["recover"])
        self.assertEqual(supervisor.status()["state"], "recovery_failed")
        self.assertFalse(supervisor.status()["busy"])
        self.assertTrue(supervisor._lock.acquire(blocking=False))
        supervisor._lock.release()

    def test_cooperative_cancel_waits_for_worker_result(self):
        context = CountingSpawnContext()
        supervisor = llm_qwen._ModelWorkerSupervisor(
            context=context, worker_target=fake_supervisor_worker, start_timeout=30.0,
        )
        cancel = Event()
        cancel.set()
        try:
            # Fake worker ignores cancellation but completes normally; ownership
            # remains held until that result arrives.
            self.assertEqual(
                supervisor.generate("normal", "", 1, 0.0, cancel, 1.0),
                "reply:normal",
            )
            self.assertEqual(supervisor.status()["state"], "ready")
        finally:
            supervisor.unload()

    def test_stream_yields_chunks_before_terminal_result(self):
        context = CountingSpawnContext()
        supervisor = llm_qwen._ModelWorkerSupervisor(
            context=context, worker_target=fake_supervisor_worker, start_timeout=30.0,
        )

        async def scenario():
            stream = supervisor.generate_stream("stream", "", 1, 0.0, None, 1.0)
            started = time.monotonic()
            first = await anext(stream)
            first_elapsed = time.monotonic() - started
            remaining = [chunk async for chunk in stream]
            return first, remaining, first_elapsed

        try:
            self.assertTrue(supervisor.load())
            first, remaining, first_elapsed = asyncio.run(scenario())
            self.assertEqual(first, "A")
            self.assertEqual(remaining, ["B"])
            self.assertLess(first_elapsed, 10.0)
            self.assertEqual(supervisor.status()["state"], "ready")
        finally:
            supervisor.unload()

    def test_stream_length_limit_propagates_after_chunks(self):
        context = CountingSpawnContext()
        supervisor = llm_qwen._ModelWorkerSupervisor(
            context=context, worker_target=fake_supervisor_worker, start_timeout=30.0,
        )
        chunks = []

        async def scenario():
            async for chunk in supervisor.generate_stream(
                "stream-length", "", 1, 0.0, None, 1.0,
            ):
                chunks.append(chunk)

        try:
            self.assertTrue(supervisor.load())
            with self.assertRaises(llm_qwen.GenerationLengthLimitError):
                asyncio.run(scenario())
            self.assertEqual(chunks, ["A"])
            self.assertEqual(llm_qwen._last_generation["stop_reason"], "length")
            self.assertEqual(supervisor.status()["state"], "ready")
        finally:
            supervisor.unload()

    def test_stream_close_cancels_and_drains_before_releasing_ownership(self):
        context = CountingSpawnContext()
        supervisor = llm_qwen._ModelWorkerSupervisor(
            context=context, worker_target=fake_supervisor_worker, start_timeout=30.0,
        )

        async def scenario():
            stream = supervisor.generate_stream(
                "stream-block", "", 1, 0.0, None, 1.0,
            )
            self.assertEqual(await anext(stream), "A")
            await stream.aclose()

        try:
            self.assertTrue(supervisor.load())
            asyncio.run(scenario())
            self.assertEqual(supervisor.status()["state"], "ready")
            self.assertEqual(
                supervisor.generate("next", "", 1, 0.0, None, 1.0),
                "reply:next",
            )
        finally:
            supervisor.unload()

    def test_stream_timeout_reaps_worker_and_readies_replacement(self):
        context = CountingSpawnContext()
        supervisor = llm_qwen._ModelWorkerSupervisor(
            context=context, worker_target=fake_supervisor_worker, start_timeout=30.0,
        )

        async def scenario():
            chunks = []
            async for chunk in supervisor.generate_stream(
                "stream-block", "", 1, 0.0, None, 0.15,
            ):
                chunks.append(chunk)
            return chunks

        try:
            self.assertTrue(supervisor.load())
            with self.assertRaises(llm_qwen.GenerationTimeoutError):
                asyncio.run(scenario())
            self.assertFalse(context.started[0].is_alive())
            deadline = time.monotonic() + 30.0
            while supervisor.status()["state"] != "ready" and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(len(context.started), 2)
            self.assertTrue(context.started[1].is_alive())
            self.assertEqual(supervisor.status()["state"], "ready")
            self.assertEqual(
                supervisor.generate("next", "", 1, 0.0, None, 1.0),
                "reply:next",
            )
        finally:
            supervisor.unload()


if __name__ == "__main__":
    unittest.main()
