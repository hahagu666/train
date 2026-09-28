import asyncio
import json
import unittest
from threading import Event
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import llm_qwen
from app.chat_orchestrator import ChatOrchestrator
from app.models import CharacterBase, SessionMeta
from app.session_manager import SessionManager
from core.parser import ParsedAction, resolve_action_targets
from core.state import CharacterState
from world.scenario_engine import get_scenario_engine
from world.world_state import WorldState


class CommandEndpointTests(unittest.TestCase):
    def test_cancel_route_sets_coordinator_cancel_event(self):
        from app import server
        from app.generation_coordinator import GenerationCoordinator

        async def scenario():
            coordinator = GenerationCoordinator()
            started = asyncio.Event()

            async def runner(cancel_event, publish):
                started.set()
                while not cancel_event.is_set():
                    await asyncio.sleep(0)
                raise RuntimeError("cancelled")

            job, _ = await coordinator.submit("session", "request", "chat", runner)
            await started.wait()
            with patch.object(server, "generation_coordinator", coordinator):
                response = await server.cancel_chat("session", "request")
            with self.assertRaises(Exception):
                await job.wait()
            await coordinator.shutdown()
            return response, job

        response, job = asyncio.run(scenario())
        self.assertTrue(response.success)
        self.assertEqual(response.data["status"], "cancelling")
        self.assertTrue(job.cancel_event.is_set())

    def test_cancel_route_is_idempotent_after_request_finishes(self):
        from app import server

        response = asyncio.run(server.cancel_chat("session", "missing"))

        self.assertTrue(response.success)
        self.assertEqual(response.data["status"], "finished")

    def test_generation_status_routes_report_jobs(self):
        from app import server
        from app.generation_coordinator import GenerationCoordinator

        async def scenario():
            coordinator = GenerationCoordinator()
            release = asyncio.Event()

            async def runner(cancel_event, publish):
                await release.wait()
                return "done"

            job, _ = await coordinator.submit("session", "request", "chat", runner)
            with patch.object(server, "generation_coordinator", coordinator):
                by_job = await server.generation_job_status(job.job_id)
                by_session = await server.session_generation_status("session")
                missing_session = await server.session_generation_status("missing")
            release.set()
            await job.wait()
            await coordinator.shutdown()
            return job, by_job, by_session, missing_session

        job, by_job, by_session, missing_session = asyncio.run(scenario())
        self.assertEqual(by_job.data["job_id"], job.job_id)
        self.assertEqual(by_session.data["request_id"], "request")
        self.assertIsNone(missing_session.data)

    def test_model_status_includes_generation_queue(self):
        from app import server

        main_status = {
            "loaded": True,
            "device": "cuda:0",
            "quantization": "4bit",
            "loading": False,
            "state": "ready",
            "busy": True,
            "error": "",
            "device_map": "cuda:0",
            "dtype": "float16",
            "model_footprint_bytes": 1,
            "cuda_memory": {},
            "attention_backend": "sdpa",
            "use_cache": True,
            "last_generation": {},
        }
        coordinator = SimpleNamespace(global_status=lambda: {
            "queue_depth": 2,
            "current_job_id": "job",
            "current_session_id": "session",
        })
        with patch.object(server, "generation_coordinator", coordinator), patch.object(
            server.llm_qwen, "get_status", return_value=main_status
        ), patch.object(server.llm_small, "is_available", return_value=False):
            response = asyncio.run(server.model_status())

        self.assertEqual(response.queue_depth, 2)
        self.assertEqual(response.current_job_id, "job")
        self.assertEqual(response.current_session_id, "session")

    def test_chat_route_is_fifo_and_duplicate_request_is_idempotent(self):
        from app import server
        from app.generation_coordinator import GenerationCoordinator
        from app.models import ChatRequest

        async def scenario():
            coordinator = GenerationCoordinator()
            first_started = asyncio.Event()
            release_first = asyncio.Event()
            calls = []

            class Orchestrator:
                async def process_message_stream(self, session_id, message, **kwargs):
                    calls.append((session_id, message))
                    if session_id == "session-a":
                        first_started.set()
                        await release_first.wait()
                    yield {"type": "state", "data": {}}
                    yield {"type": "done", "full_response": message}

            instances = {
                "session-a": SimpleNamespace(
                    messages=[],
                    build_state_snapshot=lambda: SimpleNamespace(model_dump=lambda: {}),
                ),
                "session-b": SimpleNamespace(
                    messages=[],
                    build_state_snapshot=lambda: SimpleNamespace(model_dump=lambda: {}),
                ),
            }
            manager = SimpleNamespace(
                get_session=lambda session_id: instances.get(session_id),
                get_active_session=lambda: None,
            )
            request = SimpleNamespace()
            with patch.object(server, "generation_coordinator", coordinator), patch.object(
                server, "session_mgr", manager
            ), patch.object(server, "orchestrator", Orchestrator()):
                first_task = asyncio.create_task(server.chat_nonstream(
                    ChatRequest(session_id="session-a", message="A", request_id="one"),
                    request,
                ))
                await first_started.wait()
                second_task = asyncio.create_task(server.chat_nonstream(
                    ChatRequest(session_id="session-b", message="B", request_id="two"),
                    request,
                ))
                duplicate_task = asyncio.create_task(server.chat_nonstream(
                    ChatRequest(session_id="session-b", message="ignored", request_id="two"),
                    request,
                ))
                await asyncio.sleep(0)
                queued = coordinator.get_request("session-b", "two")
                release_first.set()
                responses = await asyncio.gather(first_task, second_task, duplicate_task)
            await coordinator.shutdown()
            return responses, calls, queued

        responses, calls, queued = asyncio.run(scenario())
        self.assertEqual(calls, [("session-a", "A"), ("session-b", "B")])
        self.assertEqual(queued.request_id, "two")
        self.assertEqual(responses[1].job_id, responses[2].job_id)
        self.assertEqual(responses[1].response, "B")
        self.assertEqual(responses[2].response, "B")

    def test_browser_stream_is_ndjson_with_one_canonical_terminal_event(self):
        from app import server
        from app.generation_coordinator import GenerationCoordinator
        from app.models import ChatRequest

        async def scenario():
            coordinator = GenerationCoordinator()
            instance = SimpleNamespace(
                messages=[{"role": "assistant", "content": "AB"}],
                build_state_snapshot=lambda: SimpleNamespace(
                    model_dump=lambda: {"turn": 1}
                ),
            )
            manager = SimpleNamespace(get_session=lambda session_id: instance)

            class Orchestrator:
                async def process_message_stream(self, *args, **kwargs):
                    yield {"type": "token", "content": "A"}
                    yield {"type": "token", "content": "B"}
                    yield {"type": "state", "data": {"turn": 1}}
                    yield {"type": "done", "full_response": "AB", "snapshot_id": "snap"}

            request = ChatRequest(session_id="session", message="hello", request_id="request")
            with patch.object(server, "generation_coordinator", coordinator), patch.object(
                server, "session_mgr", manager
            ), patch.object(server, "orchestrator", Orchestrator()):
                lines = [line async for line in server._ndjson_chat_events("session", request)]
            await coordinator.shutdown()
            return lines

        lines = asyncio.run(scenario())
        self.assertTrue(all(line.endswith("\n") for line in lines))
        events = [json.loads(line) for line in lines]
        self.assertEqual([event["sequence"] for event in events], list(range(1, len(events) + 1)))
        self.assertTrue(all(event["schema_version"] == 1 for event in events))
        self.assertTrue(all(event["job_id"] == events[0]["job_id"] for event in events))
        self.assertTrue(all(event["request_id"] == "request" for event in events))
        self.assertTrue(all(event["session_id"] == "session" for event in events))
        terminals = [event for event in events if event.get("terminal")]
        self.assertEqual(len(terminals), 1)
        self.assertEqual(terminals[0]["type"], "completed")
        self.assertEqual(terminals[0]["response"], "AB")

    def test_nonstream_and_browser_stream_use_same_chat_runner(self):
        from app import server
        from app.generation_coordinator import GenerationCoordinator
        from app.models import ChatRequest

        async def scenario():
            coordinator = GenerationCoordinator()
            calls = []
            instance = SimpleNamespace(
                messages=[],
                build_state_snapshot=lambda: SimpleNamespace(model_dump=lambda: {}),
            )
            manager = SimpleNamespace(
                get_session=lambda session_id: instance,
                get_active_session=lambda: None,
            )

            class Orchestrator:
                async def process_message_stream(self, session_id, message, **kwargs):
                    calls.append((session_id, message, kwargs.get("complete_response")))
                    yield {"type": "token", "content": "shared"}
                    yield {"type": "state", "data": {}}
                    yield {"type": "done", "full_response": "shared"}

            with patch.object(server, "generation_coordinator", coordinator), patch.object(
                server, "session_mgr", manager
            ), patch.object(server, "orchestrator", Orchestrator()):
                nonstream = await server.chat_nonstream(
                    ChatRequest(session_id="session-a", message="one", request_id="request-a"),
                    SimpleNamespace(),
                )
                lines = [line async for line in server._ndjson_chat_events(
                    "session-b",
                    ChatRequest(session_id="session-b", message="two", request_id="request-b"),
                )]
            await coordinator.shutdown()
            return nonstream, [json.loads(line) for line in lines], calls

        nonstream, stream_events, calls = asyncio.run(scenario())
        self.assertEqual(nonstream.response, "shared")
        self.assertEqual(stream_events[-1]["response"], "shared")
        self.assertEqual(calls, [
            ("session-a", "one", False),
            ("session-b", "two", False),
        ])

    def test_retry_route_maps_cancelled_event_to_499(self):
        from fastapi import HTTPException
        from app import server

        instance = SimpleNamespace(
            messages=[],
            timeline_id="main",
            build_state_snapshot=lambda: SimpleNamespace(model_dump=lambda: {}),
        )
        manager = SimpleNamespace(get_session=lambda session_id: instance)

        class CancelledOrchestrator:
            async def retry_turn_stream(self, *args, **kwargs):
                yield {"type": "cancelled", "code": "CANCELLED", "message": "生成已取消"}

        request = SimpleNamespace(receive=AsyncMock())
        request.receive.side_effect = asyncio.CancelledError
        with patch.object(server, "session_mgr", manager), patch.object(
            server, "orchestrator", CancelledOrchestrator()
        ):
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(server.retry_turn(
                    "session", 1, SimpleNamespace(suggestion=None), request
                ))

        self.assertEqual(caught.exception.status_code, 499)

    def test_persistence_route_rejects_active_session_transaction(self):
        from fastapi import HTTPException
        from app import server

        manager = SimpleNamespace(
            get_session=lambda session_id: object(),
            transaction_active=lambda session_id: True,
            create_save=Mock(),
        )
        with patch.object(server, "session_mgr", manager):
            with self.assertRaises(HTTPException) as caught:
                asyncio.run(server.create_save("session", SimpleNamespace(name="存档")))

        self.assertEqual(caught.exception.status_code, 409)
        manager.create_save.assert_not_called()

    def test_commands_require_session_and_include_eligibility_metadata(self):
        from app import server

        instance = SimpleNamespace(meta=SimpleNamespace(current_stage="A"))
        manager = SimpleNamespace(get_session=lambda session_id: instance if session_id == "valid" else None)
        with patch.object(server, "session_mgr", manager):
            response = asyncio.run(server.list_chat_commands("valid"))
            self.assertTrue(response.success)
            self.assertTrue(response.data)
            self.assertTrue(all(command["min_stage"] <= "A" for command in response.data))
            self.assertTrue(all(command["group"] for command in response.data))
            with self.assertRaises(Exception) as caught:
                asyncio.run(server.list_chat_commands("missing"))
            self.assertEqual(caught.exception.status_code, 404)


class SessionTransactionTests(unittest.TestCase):
    def test_transaction_is_exclusive_and_reusable(self):
        manager = SessionManager.__new__(SessionManager)
        manager._active_transactions = set()

        self.assertTrue(manager.begin_transaction("session"))
        self.assertTrue(manager.transaction_active("session"))
        self.assertFalse(manager.begin_transaction("session"))
        manager.end_transaction("session")
        self.assertFalse(manager.transaction_active("session"))
        self.assertTrue(manager.begin_transaction("session"))

    def test_autosave_defers_without_reading_or_persisting_live_state(self):
        manager = SessionManager.__new__(SessionManager)
        manager._active_transactions = {"session"}
        instance = SimpleNamespace(save_slots={})
        manager.sessions = {"session": instance}
        manager._persist_session = Mock()

        result = manager.auto_save("session")

        self.assertEqual(result, {"status": "deferred"})
        manager._persist_session.assert_not_called()
        self.assertEqual(instance.save_slots, {})


class FakeSessionManager:
    def __init__(self, instance):
        self.instance = instance
        self.saved = False
        self.transaction = False

    def begin_transaction(self, session_id):
        if self.transaction:
            return False
        self.transaction = True
        return True

    def end_transaction(self, session_id):
        self.transaction = False

    def transaction_active(self, session_id):
        return self.transaction

    def capture_snapshot(self, instance, **kwargs):
        return {"snapshot_id": "snap"}

    def _persist_session(self, instance):
        self.saved = True

    def get_session(self, session_id):
        return self.instance

    def auto_save(self, session_id):
        if self.transaction:
            return {"status": "deferred"}
        self.saved = True

    def restore_snapshot(self, session_id, snapshot_id, new_timeline=False, persist=True):
        self.instance.char_state.relationship_trust = 0.25
        self.instance.messages = []
        self.instance.turn_count = 0
        self.instance.timeline_id = "retry-timeline" if new_timeline else "main"
        self.instance.current_snapshot_id = snapshot_id
        return self.instance


class FakeCharacterManager:
    def __init__(self, character):
        self.character = character

    def get_character(self, character_id):
        return self.character


class FakeUserManager:
    def get_profile(self):
        return SimpleNamespace(name="测试用户")

    def get_address_form(self, character_id):
        return "你"


def make_instance(character):
    state = CharacterState()
    state.relationship_trust = 0.8
    world = WorldState()
    meta = SessionMeta(
        session_id="session",
        character_id=character.id,
        character_name=character.name,
        session_name="test",
        created_at="2026-08-06T00:00:00",
        last_active_at="2026-08-06T00:00:00",
    )
    return SimpleNamespace(
        meta=meta,
        char_state=state,
        world=world,
        messages=[],
        turn_count=0,
        timeline_id="main",
        current_snapshot_id="snap-0",
        snapshot_index=[],
        conversation_summary="",
        compacted_through_turn=0,
        user_profile={"name": "事务用户", "traits": ["原始"]},
        build_state_snapshot=lambda: SimpleNamespace(model_dump=lambda: {}),
    )


class P1BackendTests(unittest.TestCase):
    def test_scenario_ids_are_unique(self):
        engine = get_scenario_engine()
        ids = [card.card_id for cards in engine.scenarios.values() for card in cards]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), len(engine.cards_by_id))

    def test_scenario_initial_state_validation_is_fieldwise(self):
        card = next(iter(get_scenario_engine().cards_by_id.values()))
        valid = card._validated_initial_state({
            "location": "bedroom", "time_hour": 21, "weekday": "Friday",
            "people_present": ["妹妹"], "emotions": {"happiness": 0.7, "bad": 1},
            "outfit": "pajamas", "trust": 0.6, "closeness": 0.5, "privacy": 0.9,
        })
        self.assertEqual(valid["outfit"], "pajamas")
        self.assertEqual(valid["trust"], 0.6)
        self.assertNotIn("bad", valid["emotions"])
        invalid = card._validated_initial_state({"outfit": "unknown", "privacy": 2, "trust": "high"})
        self.assertEqual(invalid, {})

    def test_target_resolution_does_not_mutate_clothing(self):
        state = CharacterState()
        before = state.clothing.to_dict()
        action = ParsedAction("touch_breast", ["nipple_left"])
        resolve_action_targets([action], state)
        self.assertEqual(before, state.clothing.to_dict())
        self.assertTrue(action.through_clothes)

    def test_hard_limit_blocks_before_clothing_change(self):
        character = CharacterBase(
            id="char",
            name="角色",
            limits={"soft": [], "hard": ["绝对不能让爸妈知道"]},
        )
        instance = make_instance(character)
        instance.world.parents_present = True
        orchestrator = ChatOrchestrator(
            FakeCharacterManager(character),
            FakeSessionManager(instance),
            FakeUserManager(),
        )
        action = ParsedAction(
            "handle_clothing",
            clothing_action="remove",
            clothing_target="panties",
        )
        before = instance.char_state.clothing.to_dict()
        self.assertTrue(orchestrator._blocked_by_hard_limit(
            character, action, instance.char_state, instance.world
        ))
        self.assertEqual(before, instance.char_state.clothing.to_dict())

    def test_hard_limit_does_not_block_intent_only_request(self):
        """纯成人意图（intent_only）不触发硬限制，交给角色按意愿分档回应。"""
        character = CharacterBase(
            id="char",
            name="角色",
            age=22,
            adult_verified=True,
            limits={"soft": [], "hard": ["绝对不能让爸妈知道"]},
        )
        instance = make_instance(character)
        instance.world.parents_present = True
        manager = FakeSessionManager(instance)
        orchestrator = ChatOrchestrator(
            FakeCharacterManager(character), manager, FakeUserManager()
        )

        async def collect():
            with patch(
                "app.chat_orchestrator.llm_qwen.generate", return_value="嗯……"
            ), patch(
                "app.chat_orchestrator.post_process", return_value={}
            ), patch(
                "app.chat_orchestrator.apply_post_adjustments", return_value=None
            ), patch.object(
                orchestrator, "_advance_stage", return_value="D"
            ), patch.object(
                orchestrator.memory_compactor, "compact", new=AsyncMock(return_value=None)
            ), patch.object(
                manager, "capture_snapshot", return_value={"snapshot_id": "snap"}
            ), patch.object(
                manager, "_persist_session", return_value=None
            ):
                return [event async for event in orchestrator.process_message_stream(
                    "session", "在玩肉棒，怎么了", complete_response=True
                )]

        events = asyncio.run(collect())
        self.assertFalse(any(event.get("code") == "HARD_LIMIT" for event in events))
        self.assertEqual(events[-1]["type"], "done")

    def test_repeat_detection_helpers_flag_and_pass_text(self):
        """复读检测：开头重复/整体相似被判复读，正常推进不误伤。"""
        from app.chat_orchestrator import ChatOrchestrator as CO

        recent = ["她别过头去，声音有点发抖，小声说不行。"]
        self.assertTrue(
            CO._is_duplicate_head("她别过头去，声音有点发抖，小声说了别的", recent)
        )
        self.assertFalse(
            CO._is_duplicate_head("她抬起头看着你，脸有点红，没说话", recent)
        )
        self.assertTrue(
            CO._is_repeat_of_recent("她别过头去，声音有点发抖，小声说不行。", recent)
        )
        self.assertFalse(
            CO._is_repeat_of_recent("她抿了抿嘴，半天没说话，最后小声应了一句。", recent)
        )
        self.assertEqual(CO._recent_assistant_replies(SimpleNamespace(messages=[
            {"role": "user", "content": "你说话呀"},
            {"role": "assistant", "content": "我……我在想事情。"},
        ])), ["我……我在想事情。"])
        self.assertEqual(CO._recent_assistant_replies(SimpleNamespace(messages=[])), [])

    def test_busy_error_distinguishes_recovering_state(self):
        """超时恢复期间报忙应返回 MODEL_RECOVERING，而非误导性的 MODEL_BUSY。"""
        character = CharacterBase(id="char", name="角色")
        instance = make_instance(character)
        manager = FakeSessionManager(instance)
        orchestrator = ChatOrchestrator(
            FakeCharacterManager(character), manager, FakeUserManager()
        )

        async def collect(state):
            with patch(
                "app.chat_orchestrator.parse_input", return_value=[]
            ), patch(
                "app.chat_orchestrator.llm_qwen.generate",
                side_effect=llm_qwen.ModelBusyError("模型正在生成"),
            ), patch(
                "app.chat_orchestrator.llm_qwen.get_status",
                return_value={"state": state, "busy": True},
            ):
                return [event async for event in orchestrator.process_message_stream(
                    "session", "你好", complete_response=True
                )]

        events = asyncio.run(collect("recovering"))
        self.assertEqual(events[-1]["code"], "MODEL_RECOVERING")
        events = asyncio.run(collect("generating"))
        self.assertEqual(events[-1]["code"], "MODEL_BUSY")

    def test_process_rejects_overlapping_session_transaction(self):
        character = CharacterBase(id="char", name="角色")
        instance = make_instance(character)
        manager = FakeSessionManager(instance)
        manager.transaction = True
        orchestrator = ChatOrchestrator(
            FakeCharacterManager(character), manager, FakeUserManager()
        )

        async def collect():
            return [event async for event in orchestrator.process_message_stream(
                "session", "hello", complete_response=True
            )]

        events = asyncio.run(collect())
        self.assertEqual(events, [{
            "type": "error",
            "code": "SESSION_BUSY",
            "message": "会话正在处理上一项操作",
        }])
        self.assertTrue(manager.transaction)

    def test_process_releases_transaction_after_failure(self):
        character = CharacterBase(id="char", name="角色")
        instance = make_instance(character)
        manager = FakeSessionManager(instance)
        orchestrator = ChatOrchestrator(
            FakeCharacterManager(character), manager, FakeUserManager()
        )

        async def fail_processing(*args, **kwargs):
            if False:
                yield None
            raise RuntimeError("processing failed")

        async def collect():
            with patch.object(orchestrator, "_process_message_stream", fail_processing):
                return [event async for event in orchestrator.process_message_stream(
                    "session", "hello", complete_response=True
                )]

        with self.assertRaisesRegex(RuntimeError, "processing failed"):
            asyncio.run(collect())
        self.assertFalse(manager.transaction)

    def test_failed_retry_restores_exact_pre_retry_state(self):
        character = CharacterBase(id="char", name="角色")
        instance = make_instance(character)
        instance.char_state.relationship_trust = 0.8
        instance.messages = [
            {"role": "user", "content": "原问题", "turn": 1},
            {"role": "assistant", "content": "原回答", "turn": 1},
        ]
        instance.turn_count = 1
        instance.timeline_id = "original-timeline"
        instance.current_snapshot_id = "snap-1"
        instance.snapshot_index = [
            {"turn": 0, "snapshot_id": "snap-0"},
            {"turn": 1, "snapshot_id": "snap-1"},
        ]
        manager = FakeSessionManager(instance)
        orchestrator = ChatOrchestrator(
            FakeCharacterManager(character), manager, FakeUserManager()
        )
        original_messages = [dict(message) for message in instance.messages]
        original_index = [dict(item) for item in instance.snapshot_index]

        async def fail_retry(*args, **kwargs):
            instance.char_state.relationship_trust = 0.05
            instance.messages.append({"role": "assistant", "content": "失败分支", "turn": 1})
            yield {"type": "error", "code": "LLM_TIMEOUT", "message": "生成超时"}

        async def collect():
            with patch.object(orchestrator, "_process_message_stream", fail_retry):
                return [event async for event in orchestrator.retry_turn_stream(
                    "session", 1, llm_timeout=0.01
                )]

        events = asyncio.run(collect())
        self.assertEqual(events[-1]["code"], "LLM_TIMEOUT")
        self.assertEqual(instance.char_state.relationship_trust, 0.8)
        self.assertEqual(instance.messages, original_messages)
        self.assertEqual(instance.turn_count, 1)
        self.assertEqual(instance.timeline_id, "original-timeline")
        self.assertEqual(instance.current_snapshot_id, "snap-1")
        self.assertEqual(instance.snapshot_index, original_index)
        self.assertFalse(manager.transaction)

    def test_successful_retry_keeps_new_timeline(self):
        character = CharacterBase(id="char", name="角色")
        instance = make_instance(character)
        instance.messages = [
            {"role": "user", "content": "原问题", "turn": 1},
            {"role": "assistant", "content": "原回答", "turn": 1},
        ]
        instance.turn_count = 1
        instance.snapshot_index = [{"turn": 0, "snapshot_id": "snap-0"}]
        manager = FakeSessionManager(instance)
        orchestrator = ChatOrchestrator(
            FakeCharacterManager(character), manager, FakeUserManager()
        )

        async def complete_retry(*args, **kwargs):
            instance.char_state.relationship_trust = 0.6
            instance.messages = [
                {"role": "user", "content": "原问题", "turn": 1},
                {"role": "assistant", "content": "新回答", "turn": 1},
            ]
            instance.turn_count = 1
            yield {"type": "done", "full_response": "新回答", "snapshot_id": "snap-new"}

        async def collect():
            with patch.object(orchestrator, "_process_message_stream", complete_retry):
                return [event async for event in orchestrator.retry_turn_stream("session", 1)]

        events = asyncio.run(collect())
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(instance.timeline_id, "retry-timeline")
        self.assertEqual(instance.messages[-1]["content"], "新回答")
        self.assertEqual(instance.char_state.relationship_trust, 0.6)
        self.assertFalse(manager.transaction)

    def test_orchestrator_passes_parsed_action_type_to_post_processor(self):
        character = CharacterBase(id="char", name="角色")
        instance = make_instance(character)
        manager = FakeSessionManager(instance)
        orchestrator = ChatOrchestrator(FakeCharacterManager(character), manager, FakeUserManager())
        post = Mock(return_value={})

        async def collect():
            with patch(
                "app.chat_orchestrator.parse_input",
                return_value=[ParsedAction(action_type="talk", verbal="你好")],
            ), patch(
                "app.chat_orchestrator.llm_qwen.generate", return_value="你好"
            ), patch(
                "app.chat_orchestrator.post_process", post
            ), patch(
                "app.chat_orchestrator.apply_post_adjustments", return_value=None
            ), patch.object(
                orchestrator, "_advance_stage", return_value="D"
            ), patch.object(
                orchestrator.memory_compactor, "compact", new=AsyncMock(return_value=None)
            ), patch.object(
                manager, "capture_snapshot", return_value={"snapshot_id": "snap"}
            ), patch.object(
                manager, "_persist_session", return_value=None
            ):
                return [event async for event in orchestrator.process_message_stream(
                    "session", "我笑着说：你好", complete_response=True
                )]

        events = asyncio.run(collect())

        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(post.call_args.args[3], "talk")

    def test_length_limited_response_rolls_back_without_persisting_partial_text(self):
        character = CharacterBase(id="char", name="角色")
        instance = make_instance(character)
        manager = FakeSessionManager(instance)
        orchestrator = ChatOrchestrator(FakeCharacterManager(character), manager, FakeUserManager())

        async def limited_stream(*args, **kwargs):
            yield "被截断的回复"
            raise llm_qwen.GenerationLengthLimitError("length limit")

        async def collect():
            with patch("app.chat_orchestrator.parse_input", return_value=[]), patch(
                "app.chat_orchestrator.llm_qwen.generate_stream", side_effect=limited_stream
            ):
                return [event async for event in orchestrator.process_message_stream(
                    "session", "很多动作", complete_response=False
                )]

        events = asyncio.run(collect())
        self.assertTrue(any(event["type"] == "token" for event in events))
        self.assertEqual(events[-1]["code"], "LLM_LENGTH_LIMIT")
        self.assertEqual(instance.turn_count, 0)
        self.assertEqual(instance.messages, [])
        self.assertFalse(manager.saved)

    def test_complete_response_does_not_emit_tokens(self):
        character = CharacterBase(id="char", name="角色")
        instance = make_instance(character)
        manager = FakeSessionManager(instance)
        orchestrator = ChatOrchestrator(FakeCharacterManager(character), manager, FakeUserManager())

        async def collect():
            with patch("app.chat_orchestrator.parse_input", return_value=[]), patch(
                "app.chat_orchestrator.llm_qwen.generate", return_value="你好"
            ), patch("app.chat_orchestrator.post_process", return_value={}), patch(
                "app.chat_orchestrator.apply_post_adjustments", return_value=None
            ), patch.object(orchestrator, "_advance_stage", return_value="D"), patch.object(
                orchestrator.memory_compactor, "compact", new=AsyncMock(return_value=None)
            ), patch.object(manager, "capture_snapshot", return_value={"snapshot_id": "snap"}), patch.object(
                manager, "_persist_session", return_value=None
            ):
                return [event async for event in orchestrator.process_message_stream(
                    "session", "hello", complete_response=True
                )]

        events = asyncio.run(collect())
        self.assertFalse(any(event["type"] == "token" for event in events))
        self.assertEqual(events[-1]["type"], "done")
        self.assertEqual(events[-1]["full_response"], "你好")

    def test_post_process_failure_rolls_back_turn(self):
        character = CharacterBase(id="char", name="角色")
        instance = make_instance(character)
        manager = FakeSessionManager(instance)
        orchestrator = ChatOrchestrator(FakeCharacterManager(character), manager, FakeUserManager())

        def fail_post_process(*args, **kwargs):
            instance.char_state.relationship_trust = 0.1
            raise RuntimeError("post process failed")

        async def collect():
            with patch("app.chat_orchestrator.parse_input", return_value=[]), patch(
                "app.chat_orchestrator.llm_qwen.generate", return_value="你好"
            ), patch("app.chat_orchestrator.post_process", side_effect=fail_post_process):
                return [event async for event in orchestrator.process_message_stream(
                    "session", "hello", complete_response=True
                )]

        events = asyncio.run(collect())
        self.assertEqual(events[-1]["code"], "POST_PROCESS_FAILED")
        self.assertEqual(instance.char_state.relationship_trust, 0.8)
        self.assertEqual(instance.turn_count, 0)
        self.assertEqual(instance.messages, [])
        self.assertFalse(manager.saved)

    def test_commit_failure_rolls_back_turn(self):
        character = CharacterBase(id="char", name="角色")
        instance = make_instance(character)
        manager = FakeSessionManager(instance)
        orchestrator = ChatOrchestrator(FakeCharacterManager(character), manager, FakeUserManager())

        async def collect():
            with patch("app.chat_orchestrator.parse_input", return_value=[]), patch(
                "app.chat_orchestrator.llm_qwen.generate", return_value="你好"
            ), patch("app.chat_orchestrator.post_process", return_value={}), patch(
                "app.chat_orchestrator.apply_post_adjustments", return_value=None
            ), patch.object(orchestrator, "_advance_stage", return_value="D"), patch.object(
                orchestrator.memory_compactor, "compact", new=AsyncMock(return_value=None)
            ), patch.object(manager, "capture_snapshot", side_effect=OSError("disk full")):
                return [event async for event in orchestrator.process_message_stream(
                    "session", "hello", complete_response=True
                )]

        events = asyncio.run(collect())
        self.assertEqual(events[-1]["code"], "COMMIT_FAILED")
        self.assertEqual(instance.char_state.relationship_trust, 0.8)
        self.assertEqual(instance.turn_count, 0)
        self.assertEqual(instance.messages, [])
        self.assertFalse(manager.saved)

    def test_cancelled_turn_during_post_process_rolls_back_and_is_not_saved(self):
        character = CharacterBase(id="char", name="角色")
        instance = make_instance(character)
        manager = FakeSessionManager(instance)
        orchestrator = ChatOrchestrator(
            FakeCharacterManager(character), manager, FakeUserManager()
        )
        cancel = Event()
        post_started = Event()
        release_post = Event()

        def slow_post_process(*args, **kwargs):
            instance.char_state.relationship_trust = 0.1
            instance.user_profile["name"] = "瞬态用户"
            instance.user_profile["traits"].append("瞬态")
            post_started.set()
            release_post.wait(timeout=2)
            return {"relationship_trust": 0.1}

        async def collect():
            async def cancel_during_post_process():
                await asyncio.to_thread(post_started.wait, 2)
                cancel.set()
                release_post.set()

            with patch("app.chat_orchestrator.parse_input", return_value=[]), patch(
                "app.chat_orchestrator.llm_qwen.generate", return_value="你好"
            ), patch(
                "app.chat_orchestrator.post_process", side_effect=slow_post_process
            ), patch(
                "app.chat_orchestrator.apply_post_adjustments"
            ) as apply_adjustments:
                cancel_task = asyncio.create_task(cancel_during_post_process())
                events = [event async for event in orchestrator.process_message_stream(
                    "session", "hello", cancel_event=cancel, complete_response=True
                )]
                await cancel_task
                return events, apply_adjustments

        events, apply_adjustments = asyncio.run(collect())
        self.assertEqual(events[-1]["type"], "cancelled")
        self.assertEqual(instance.char_state.relationship_trust, 0.8)
        self.assertEqual(instance.turn_count, 0)
        self.assertEqual(instance.messages, [])
        self.assertEqual(instance.user_profile, {"name": "事务用户", "traits": ["原始"]})
        self.assertFalse(manager.saved)
        apply_adjustments.assert_not_called()

    def test_cancelled_turn_rolls_back_and_is_not_saved(self):
        character = CharacterBase(id="char", name="角色")
        instance = make_instance(character)
        manager = FakeSessionManager(instance)
        orchestrator = ChatOrchestrator(
            FakeCharacterManager(character), manager, FakeUserManager()
        )
        cancel = Event()

        async def fake_stream(*args, **kwargs):
            instance.char_state.relationship_trust = 0.1
            instance.user_profile["name"] = "瞬态用户"
            cancel.set()
            yield "partial"

        async def collect():
            with patch("app.chat_orchestrator.parse_input", return_value=[]), patch(
                "app.chat_orchestrator.llm_qwen.generate_stream", fake_stream
            ):
                return [event async for event in orchestrator.process_message_stream(
                    "session", "hello", cancel_event=cancel
                )]

        events = asyncio.run(collect())
        self.assertEqual(events[-1]["type"], "cancelled")
        self.assertEqual(instance.char_state.relationship_trust, 0.8)
        self.assertEqual(instance.turn_count, 0)
        self.assertEqual(instance.messages, [])
        self.assertEqual(instance.user_profile, {"name": "事务用户", "traits": ["原始"]})
        self.assertFalse(manager.saved)


if __name__ == "__main__":
    unittest.main()
