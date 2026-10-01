"""
FastAPI 后端服务入口
提供 REST API + WebSocket 流式对话接口
"""
import os
import sys
import asyncio
import json
import time
from copy import deepcopy
from datetime import datetime
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.exceptions import RequestValidationError

# 添加项目根目录到sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.logger import info, success, warning, error, debug, separator
from app.config import SERVER_HOST, SERVER_PORT, CORS_ORIGINS
from app.models import (
    StandardResponse, ModelStatusResponse,
    UserProfileResponse, SetUserProfileRequest, UpdateUserNameRequest, SetRelationRequest,
    CharacterSummary, CreateCharacterRequest, UpdateCharacterRequest,
    SessionMeta, CreateSessionRequest, RenameSessionRequest,
    ChatRequest, ChatResponse, SessionStateSnapshot,
    SaveSlotSummary, CreateSaveRequest, BranchRequest, LoadSaveRequest,
    RollbackRequest, TurnOperationRequest, UpdateAppSettingsRequest,
    CreateKnowledgeRequest,
)
from app.character_manager import CharacterManager
from app.user_profile import UserProfileManager
from app.session_manager import SessionManager
from core.serialization import serialize_state
from app.chat_orchestrator import ChatOrchestrator
from app.generation_coordinator import (
    GenerationCoordinator,
    GenerationJobCancelledError,
    SessionGenerationBusyError,
)
from app.app_settings import AppSettingsManager
# 推理后端路由：llama(GGUF) 或 transformers，对外仍以 llm_qwen 名称调用
from app import model_backend as llm_qwen
import llm_small


# === 全局服务实例 ===
char_mgr: Optional[CharacterManager] = None
user_mgr: Optional[UserProfileManager] = None
session_mgr: Optional[SessionManager] = None
knowledge_svc = None
orchestrator: Optional[ChatOrchestrator] = None
app_settings_mgr = AppSettingsManager()
model_loading_task = None
server_start_time = time.time()
generation_coordinator = GenerationCoordinator()


class GenerationEventError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _ensure_session_writable(session_id: str):
    """拒绝在聊天或重说事务期间读取瞬态状态执行其他写操作。"""
    if session_mgr and session_mgr.transaction_active(session_id):
        raise HTTPException(409, "会话正在处理上一项操作")


def _make_async_small_llm():
    """把同步的llm_small.generate包装成async函数，避免阻塞事件循环"""
    import functools
    loop = asyncio.get_event_loop()

    async def _async_generate(prompt: str, max_new_tokens: int = 200, temperature: float = 0.3, **kw) -> str:
        fn = functools.partial(llm_small.generate, prompt,
                               max_new_tokens=max_new_tokens, temperature=temperature, **kw)
        return await loop.run_in_executor(None, fn)

    return _async_generate


@asynccontextmanager
async def lifespan(app: FastAPI):
    """服务启动/关闭生命周期"""
    global char_mgr, user_mgr, session_mgr, knowledge_svc, orchestrator, model_loading_task
    
    separator("═")
    info("服务器", "正在初始化后端服务...")
    
    # 小模型async wrapper（按需加载到CPU，不阻塞启动）
    small_available = llm_small.is_available()
    info("服务器", f"小模型检测: {'可用' if small_available else '未找到，使用降级模式'}")
    small_llm = _make_async_small_llm() if small_available else None
    
    # 初始化管理器
    info("服务器", "正在初始化角色管理器...")
    char_mgr = CharacterManager(small_llm_fn=small_llm)
    success("服务器", "✓ 角色管理器初始化完成")
    debug("角色", f"预设角色数: {len(char_mgr.list_presets())}")
    
    info("服务器", "正在初始化用户人设管理器...")
    user_mgr = UserProfileManager(small_llm_fn=small_llm)
    success("服务器", "✓ 用户人设管理器初始化完成")
    
    info("服务器", "正在初始化知识库服务...")
    from app.knowledge_service import KnowledgeService
    knowledge_svc = KnowledgeService()
    success("服务器", "✓ 知识库服务初始化完成")

    info("服务器", "正在初始化会话管理器...")
    session_mgr = SessionManager(character_manager=char_mgr, knowledge_service=knowledge_svc)
    success("服务器", "✓ 会话管理器初始化完成")

    info("服务器", "正在初始化对话编排器...")
    orchestrator = ChatOrchestrator(
        char_mgr=char_mgr, session_mgr=session_mgr,
        user_mgr=user_mgr, knowledge_svc=knowledge_svc,
        small_llm_fn=small_llm, settings_mgr=app_settings_mgr,
    )
    success("服务器", "✓ 对话编排器初始化完成")
    
    # 后台异步加载主模型
    async def load_model_async():
        info("大模型", "开始后台加载主模型 (Qwen2.5-7B-Instruct)...")
        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(None, llm_qwen.load_model)
        except Exception as e:
            error("大模型", f"模型加载失败: {e}", exc_info=True)
    model_loading_task = asyncio.create_task(load_model_async())
    
    success("服务器", f"✓ API服务启动在 http://{SERVER_HOST}:{SERVER_PORT}")
    info("服务器", f"  API文档: http://{SERVER_HOST}:{SERVER_PORT}/docs")
    info("服务器", f"  WebSocket: ws://{SERVER_HOST}:{SERVER_PORT}/ws/chat/{{session_id}}")
    separator("─")
    
    yield
    
    # 关闭时清理
    info("服务器", "正在关闭服务...")
    await generation_coordinator.shutdown()
    if model_loading_task and not model_loading_task.done():
        model_loading_task.cancel()
        debug("服务器", "已停止等待模型加载；后台加载线程会自行结束")
    try:
        llm_qwen.unload_model()
    except Exception as exc:
        warning("大模型", f"关闭模型工作进程失败: {exc}")
    llm_small.unload()
    success("服务器", "✓ 服务已正常关闭")


app = FastAPI(title="Heart Chat Backend", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ====== 请求日志中间件 ======
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    path = request.url.path
    method = request.method
    
    # 健康检查不打debug日志，避免刷屏
    is_health = path == "/api/health"
    
    if not is_health:
        debug("接口", f"{method} {path}")
    
    try:
        response = await call_next(request)
        elapsed = (time.time() - start) * 1000
        
        if not is_health:
            status_color = "success" if response.status_code < 400 else "warning" if response.status_code < 500 else "error"
            resp_line = f"  → {response.status_code} {method} {path} ({elapsed:.1f}ms)"
            if status_color == "success":
                debug("接口", resp_line)
            elif status_color == "warning":
                warning("接口", resp_line)
            else:
                error("接口", resp_line)
        return response
    except Exception as e:
        elapsed = (time.time() - start) * 1000
        error("接口", f"{method} {path} → 异常 ({elapsed:.1f}ms): {e}", exc_info=True)
        raise


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    code = {
        400: "INVALID_REQUEST",
        404: "NOT_FOUND",
        409: "CONFLICT",
    }.get(exc.status_code, "HTTP_ERROR")
    # 记录详细定位：方法+路径+参数+错误详情
    detail = str(exc.detail)
    qs = str(request.url.query)
    loc = f"{request.method} {request.url.path}" + (f"?{qs}" if qs else "")
    if exc.status_code >= 500:
        error("接口", f"HTTP {exc.status_code} [{code}] {loc} → {detail}")
    elif exc.status_code >= 400:
        warning("接口", f"HTTP {exc.status_code} [{code}] {loc} → {detail}")
    return JSONResponse(status_code=exc.status_code, content={
        "success": False,
        "code": code,
        "message": detail,
        "data": None,
    })


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content={
        "success": False, "code": "VALIDATION_ERROR", "message": "请求参数无效",
        "data": exc.errors(),
    })


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=400, content={
        "success": False, "code": "INVALID_REQUEST", "message": str(exc), "data": None,
    })


@app.get("/api/health")
async def health():
    loaded = llm_qwen.is_loaded()
    return {"status": "ok" if loaded else "loading", "service": "heart-chat", "model_loaded": loaded}


@app.get("/api/model/status", response_model=ModelStatusResponse)
async def model_status():
    main = llm_qwen.get_status()
    queue = generation_coordinator.global_status()
    small_loaded = bool(getattr(llm_small, "_model", None))
    return ModelStatusResponse(
        main_model_loaded=main["loaded"],
        main_model_device=main["device"],
        main_model_quantization=main["quantization"],
        small_model_loaded=small_loaded,
        small_model_available=llm_small.is_available(),
        loading=main["loading"],
        main_model_state=main["state"],
        main_model_busy=main["busy"],
        main_model_error=main["error"],
        main_model_device_map=main["device_map"],
        main_model_dtype=main["dtype"],
        main_model_footprint_bytes=main["model_footprint_bytes"],
        cuda_memory=main["cuda_memory"],
        attention_backend=main["attention_backend"],
        use_cache=main["use_cache"],
        last_generation=main["last_generation"],
        queue_depth=queue["queue_depth"],
        current_job_id=queue["current_job_id"],
        current_session_id=queue["current_session_id"],
    )


def _model_loaded_flag():
    """检查主模型和小模型加载状态（辅助函数）"""
    main_loaded = llm_qwen.is_loaded()
    small_loaded = getattr(llm_small, '_model', None) is not None
    return main_loaded, small_loaded


@app.post("/api/model/preload")
async def preload_model():
    if not llm_qwen.is_loaded():
        info("大模型", "收到预加载请求...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, llm_qwen.load_model)
    return {"success": True, "loaded": llm_qwen.is_loaded()}


@app.post("/api/model/unload")
async def unload_model():
    try:
        llm_qwen.unload_model()
    except llm_qwen.ModelBusyError as exc:
        raise HTTPException(409, str(exc))
    return StandardResponse(success=True)


@app.post("/api/model/reload")
async def reload_model():
    try:
        loop = asyncio.get_event_loop()
        loaded = await loop.run_in_executor(None, llm_qwen.reload_model)
    except llm_qwen.ModelBusyError as exc:
        raise HTTPException(409, str(exc))
    if not loaded:
        raise HTTPException(503, "模型加载失败")
    return StandardResponse(success=True)


# ====== 用户人设 ======
@app.get("/api/user/profile", response_model=UserProfileResponse)
async def get_user_profile():
    debug("用户", "获取用户人设")
    return user_mgr.get_profile()


@app.put("/api/user/profile", response_model=StandardResponse)
async def set_user_profile(req: SetUserProfileRequest):
    info("用户", "更新用户人设")
    debug("用户", f"人设描述长度: {len(req.description)}字")
    profile = await user_mgr.set_profile_from_text(req.description)
    success("用户", "✓ 人设已更新")
    return StandardResponse(success=True, message="人设已更新", data=profile.model_dump())


@app.patch("/api/user/profile/name")
async def update_user_name(req: UpdateUserNameRequest):
    info("用户", f"更新用户名字: {req.name}")
    user_mgr.update_name(req.name)
    return StandardResponse(success=True, message="名字已更新")


@app.post("/api/user/profile/relation")
async def set_relation(req: SetRelationRequest):
    info("用户", f"设置角色关系: char={req.character_id}, relation={req.relation}")
    user_mgr.set_relation(req.character_id, req.relation, req.custom_address)
    success("用户", "✓ 关系已设置")
    return StandardResponse(success=True, message="关系已设置")


# ====== 角色管理 ======
@app.get("/api/characters", response_model=list[CharacterSummary])
async def list_characters():
    chars = char_mgr.list_characters()
    debug("角色", f"获取角色列表: {len(chars)}个")
    return chars


@app.get("/api/characters/presets", response_model=list[dict])
async def list_presets():
    presets = char_mgr.list_presets()
    debug("角色", f"获取预设角色列表: {len(presets)}个")
    return [c.model_dump() for c in presets]


@app.get("/api/characters/{char_id}/eligibility")
async def get_character_eligibility(char_id: str):
    from core.content_policy import evaluate_character

    c = char_mgr.get_character(char_id)
    if not c:
        raise HTTPException(404, "角色不存在")
    decision = evaluate_character(c)
    return StandardResponse(success=True, data={
        "character_id": c.id,
        "age": c.age,
        "adult_verified": c.adult_verified,
        "sexual_interaction_allowed": decision.allowed,
        "code": decision.code,
        "reason": decision.reason,
        "eligibility_version": decision.version,
    })


@app.get("/api/characters/{char_id}")
async def get_character(char_id: str):
    debug("角色", f"获取角色详情: {char_id}")
    c = char_mgr.get_character(char_id)
    if not c:
        warning("角色", f"角色不存在: {char_id}")
        raise HTTPException(404, "角色不存在")
    return c.model_dump()


@app.post("/api/characters")
async def create_character(req: CreateCharacterRequest):
    info("角色", f"创建新角色: {req.name}")
    char = char_mgr.create_character(req)
    success("角色", f"✓ 角色创建成功: {char.name} (ID: {char.id})")
    return StandardResponse(success=True, data=char.model_dump())


@app.post("/api/characters/{char_id}/media/{kind}")
async def upload_character_media(
    char_id: str,
    kind: str,
    file: UploadFile = File(...),
    crop_x: float = Form(...),
    crop_y: float = Form(...),
    crop_width: float = Form(...),
    crop_height: float = Form(...),
):
    character = char_mgr.get_character(char_id)
    if not character:
        raise HTTPException(404, "角色不存在")
    payload = await file.read(char_mgr.media_service.MAX_BYTES + 1)
    result = char_mgr.media_service.process_upload(
        char_id, kind, payload, (crop_x, crop_y, crop_width, crop_height)
    )
    char_mgr.media_service.apply_to_character(character)
    if not character.is_preset:
        char_mgr._save_custom_character(character)
    return StandardResponse(success=True, data=result)


@app.get("/api/media/{char_id}/{kind}")
async def get_character_media(char_id: str, kind: str):
    if not char_mgr.get_character(char_id):
        raise HTTPException(404, "角色不存在")
    entry = char_mgr.media_service.get_entry(char_id, kind)
    if not entry:
        raise HTTPException(404, "角色媒体不存在")
    return FileResponse(entry["path"], media_type="image/jpeg")


@app.patch("/api/characters/{char_id}")
async def update_character(char_id: str, req: UpdateCharacterRequest):
    info("角色", f"更新角色: {char_id}")
    try:
        c = char_mgr.update_character(char_id, req)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not c:
        warning("角色", f"更新失败，角色不存在: {char_id}")
        raise HTTPException(404, "角色不存在")
    success("角色", f"✓ 角色已更新: {char_id}")
    return StandardResponse(success=True, data=c.model_dump())


@app.delete("/api/characters/{char_id}")
async def delete_character(char_id: str):
    info("角色", f"删除角色: {char_id}")
    ok = char_mgr.delete_character(char_id)
    if not ok:
        warning("角色", f"删除失败，无法删除预设角色或角色不存在: {char_id}")
        raise HTTPException(400, "无法删除预设角色或角色不存在")
    success("角色", f"✓ 角色已删除: {char_id}")
    return StandardResponse(success=True)


@app.get("/api/knowledge/shared")
async def list_shared_knowledge(limit: int = 50, cursor: Optional[str] = None):
    try:
        items, next_cursor = knowledge_svc.list_memories_page("shared", limit=limit, cursor=cursor)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"items": [item.to_dict() for item in items], "next_cursor": next_cursor, "has_more": next_cursor is not None}


@app.post("/api/knowledge/shared")
async def create_shared_knowledge(req: CreateKnowledgeRequest):
    item = knowledge_svc.create_api_memory("shared", req)
    return StandardResponse(success=True, data=item.to_dict())


@app.get("/api/knowledge/character/{char_id}")
async def list_character_knowledge(char_id: str, limit: int = 50, cursor: Optional[str] = None):
    if not char_mgr.get_character(char_id):
        raise HTTPException(404, "角色不存在")
    try:
        items, next_cursor = knowledge_svc.list_memories_page("character_private", char_id=char_id, limit=limit, cursor=cursor)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"items": [item.to_dict() for item in items], "next_cursor": next_cursor, "has_more": next_cursor is not None}


@app.post("/api/knowledge/character/{char_id}")
async def create_character_knowledge(char_id: str, req: CreateKnowledgeRequest):
    if not char_mgr.get_character(char_id):
        raise HTTPException(404, "角色不存在")
    item = knowledge_svc.create_api_memory("character_private", req, char_id=char_id)
    return StandardResponse(success=True, data=item.to_dict())


@app.get("/api/knowledge/session/{session_id}")
async def list_session_knowledge(session_id: str, limit: int = 50, cursor: Optional[str] = None):
    inst = session_mgr.get_session(session_id)
    if not inst:
        raise HTTPException(404, "会话不存在")
    try:
        items, next_cursor = knowledge_svc.list_memories_page("relationship", session_id=session_id, limit=limit, cursor=cursor)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"items": [item.to_dict() for item in items], "next_cursor": next_cursor, "has_more": next_cursor is not None}


@app.post("/api/knowledge/session/{session_id}")
async def create_session_knowledge(session_id: str, req: CreateKnowledgeRequest):
    inst = session_mgr.get_session(session_id)
    if not inst:
        raise HTTPException(404, "会话不存在")
    item = knowledge_svc.create_api_memory("relationship", req, char_id=inst.meta.character_id, session_id=session_id)
    return StandardResponse(success=True, data=item.to_dict())


# ====== 会话管理 ======
@app.get("/api/app/menu")
async def app_menu():
    return StandardResponse(success=True, data={
        "characters": [c.model_dump() for c in char_mgr.list_characters()],
        "presets": [c.model_dump() for c in char_mgr.list_presets()],
        "sessions": [m.model_dump() for m in session_mgr.list_sessions()],
        "active_session_id": session_mgr.active_session_id,
        "settings": app_settings_mgr.get().model_dump(),
    })


@app.get("/api/app/about")
async def app_about():
    return StandardResponse(success=True, data={
        "name": "Sister Train",
        "version": "0.1.0",
        "backend": "FastAPI",
    })


@app.get("/api/app/settings")
async def get_app_settings():
    return StandardResponse(success=True, data=app_settings_mgr.get().model_dump())


@app.patch("/api/app/settings")
async def update_app_settings(req: UpdateAppSettingsRequest):
    return StandardResponse(success=True, data=app_settings_mgr.update(req).model_dump())


@app.get("/api/sessions/corrupted")
async def list_corrupted_sessions():
    return StandardResponse(success=True, data=session_mgr.list_corrupted())


@app.get("/api/sessions", response_model=list[SessionMeta])
async def list_sessions(char_id: Optional[str] = None):
    sessions = session_mgr.list_sessions(char_id)
    debug("会话", f"获取会话列表: {len(sessions)}个")
    return sessions


@app.get("/api/scenarios")
async def list_scenarios():
    from world.scenario_engine import get_scenario_engine
    return get_scenario_engine().list_intro_summaries()


@app.get("/api/chat/commands")
async def list_chat_commands(session_id: str):
    from core.parser import get_command_catalog

    inst = session_mgr.get_session(session_id) if session_mgr else None
    if not inst:
        raise HTTPException(404, "会话不存在")
    current_stage = inst.meta.current_stage or "A"
    commands = [
        {
            "id": command["id"],
            "label": command["label"],
            "description": f"发送“{command['text']}”并交由场景解析器处理",
            "content": command["text"],
            "group": command["group"],
            "min_stage": command["min_stage"],
        }
        for command in get_command_catalog()
        if current_stage >= command["min_stage"]
    ]
    return StandardResponse(success=True, data=commands)


@app.post("/api/sessions")
async def create_session(req: CreateSessionRequest):
    info("会话", f"创建新会话: char={req.character_id}, name={req.name}, scenario={req.scenario_id}")
    if req.scenario_id:
        from world.scenario_engine import get_scenario_engine
        if not get_scenario_engine().get_card(req.scenario_id):
            raise HTTPException(422, "剧情卡不存在")
    if not char_mgr or not char_mgr.get_character(req.character_id):
        raise HTTPException(404, "角色不存在")
    try:
        char_obj = char_mgr.get_character(req.character_id) if char_mgr else None
        session_profile = user_mgr.build_session_snapshot(req.character_id, req.user_profile, char=char_obj)
        inst = session_mgr.create_session(
            req.character_id,
            req.name,
            req.scenario_id,
            custom_opening=req.custom_opening,
            initial_trust=req.initial_trust,
            initial_closeness=req.initial_closeness,
            initial_outfit=req.initial_outfit,
            custom_outfit_description=req.custom_outfit_description,
            location=req.location,
            time_str=req.time_str,
            privacy=req.privacy,
            user_profile=session_profile,
        )
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    success("会话", f"✓ 会话创建成功: {inst.meta.session_id}")
    opening_message = inst.get_opening_message()
    opening = opening_message.get("content", "") if opening_message else ""
    if opening:
        debug("会话", f"开场文本长度: {len(opening)}字")
    return StandardResponse(success=True, data={
        "session_id": inst.meta.session_id,
        "meta": inst.meta.model_dump(),
        "opening": opening,
        "opening_message": opening_message,
        "scenario_id": inst.scenario_id,
        "state": inst.build_state_snapshot().model_dump(),
        "snapshot_id": inst.current_snapshot_id,
        "timeline_id": inst.timeline_id,
        "user_profile": deepcopy(inst.user_profile),
    })


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    debug("会话", f"获取会话详情: {session_id}")
    inst = session_mgr.get_session(session_id)
    if not inst:
        warning("会话", f"会话不存在: {session_id}")
        raise HTTPException(404, "会话不存在")
    snap = inst.build_state_snapshot()
    return {
        "meta": inst.meta.model_dump(),
        "state": snap.model_dump(),
        "messages": inst.messages,
        "scenario_id": inst.scenario_id,
        "snapshot_id": inst.current_snapshot_id,
        "timeline_id": inst.timeline_id,
        "user_profile": deepcopy(inst.user_profile),
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    info("会话", f"删除会话: {session_id}")
    _ensure_session_writable(session_id)
    ok = session_mgr.delete_session(session_id)
    if not ok:
        warning("会话", f"删除失败，会话不存在: {session_id}")
        raise HTTPException(404, "会话不存在")
    success("会话", f"✓ 会话已删除: {session_id}")
    return StandardResponse(success=True)


@app.patch("/api/sessions/{session_id}/name")
async def rename_session(session_id: str, req: RenameSessionRequest):
    info("会话", f"重命名会话: {session_id} → {req.name}")
    _ensure_session_writable(session_id)
    if not session_mgr.rename_session(session_id, req.name):
        raise HTTPException(404, "会话不存在")
    return StandardResponse(success=True)


@app.post("/api/sessions/{session_id}/switch")
async def switch_session(session_id: str):
    info("会话", f"切换活跃会话: {session_id}")
    inst = session_mgr.switch_session(session_id)
    if not inst:
        warning("会话", f"切换失败，会话不存在: {session_id}")
        raise HTTPException(404, "会话不存在")
    success("会话", f"✓ 已切换到会话: {session_id}")
    return StandardResponse(success=True)


@app.post("/api/sessions/{session_id}/reset")
async def reset_session(session_id: str):
    info("会话", f"重置会话: {session_id}")
    _ensure_session_writable(session_id)
    inst = session_mgr.get_session(session_id)
    if not inst:
        warning("会话", f"重置失败，会话不存在: {session_id}")
        raise HTTPException(404, "会话不存在")
    char_id = inst.meta.character_id
    name = inst.meta.session_name
    scenario_id = inst.scenario_id
    profile_snapshot = deepcopy(inst.user_profile)
    session_mgr.delete_session(session_id)
    new_inst = session_mgr.create_session(
        char_id, name, scenario_id, user_profile=profile_snapshot,
    )
    success("会话", f"✓ 会话已重置，新ID: {new_inst.meta.session_id}")
    return StandardResponse(success=True, data={"new_session_id": new_inst.meta.session_id})


@app.put("/api/sessions/{session_id}/autosave")
async def autosave_session(session_id: str):
    autosave = session_mgr.auto_save(session_id)
    if autosave is None:
        raise HTTPException(404, "会话不存在")
    return StandardResponse(success=True, data={"_autosave": autosave})


@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    inst = session_mgr.get_session(session_id)
    if not inst:
        raise HTTPException(404, "会话不存在")
    return StandardResponse(success=True, data={
        "messages": inst.messages,
        "snapshots": session_mgr.list_history(session_id),
        "current_snapshot_id": inst.current_snapshot_id,
        "timeline_id": inst.timeline_id,
    })


@app.get("/api/sessions/{session_id}/snapshots")
async def list_snapshots(session_id: str):
    if not session_mgr.get_session(session_id):
        raise HTTPException(404, "会话不存在")
    return StandardResponse(success=True, data=session_mgr.list_history(session_id))


@app.get("/api/sessions/{session_id}/snapshots/{snapshot_id}")
async def get_snapshot(session_id: str, snapshot_id: str):
    snapshot = session_mgr.get_snapshot(session_id, snapshot_id)
    if not snapshot:
        raise HTTPException(404, "快照不存在")
    return StandardResponse(success=True, data=snapshot)


@app.get("/api/sessions/{session_id}/state")
async def get_session_state(session_id: str):
    inst = session_mgr.get_session(session_id)
    if not inst:
        raise HTTPException(404, "会话不存在")
    return StandardResponse(success=True, data={
        "state": inst.build_state_snapshot().model_dump(),
        "character": serialize_state(inst.char_state, inst.world, getattr(inst.char_state, "memory", None)),
        "world": inst.world.to_dict(),
        "snapshot_id": inst.current_snapshot_id,
        "timeline_id": inst.timeline_id,
    })


@app.post("/api/sessions/{session_id}/retry")
async def retry_turn(session_id: str, turn: int, req: TurnOperationRequest, request: Request):
    if not session_mgr.get_session(session_id):
        raise HTTPException(404, "会话不存在")

    async def runner(cancel_event, publish):
        full = ""
        state_snap = None
        snapshot_id = None
        terminal_error = None
        async for event in orchestrator.retry_turn_stream(
                session_id, turn, req.suggestion or "", cancel_event=cancel_event):
            await publish(event)
            if event["type"] == "token":
                full += event["content"]
            elif event["type"] == "state":
                state_snap = event["data"]
                snapshot_id = event.get("snapshot_id")
            elif event["type"] == "done":
                full = event["full_response"]
                snapshot_id = event.get("snapshot_id", snapshot_id)
            elif event["type"] in {"error", "cancelled"}:
                terminal_error = event
        if terminal_error:
            code = terminal_error.get("code", "LLM_ERROR")
            message = terminal_error.get("message", "重说失败")
            if code == "CANCELLED":
                raise GenerationJobCancelledError(message)
            raise GenerationEventError(code, message)
        inst = session_mgr.get_session(session_id)
        return {
            "response": full,
            "messages": inst.messages,
            "state": state_snap or inst.build_state_snapshot().model_dump(),
            "snapshot_id": snapshot_id,
            "timeline_id": inst.timeline_id,
        }

    try:
        job, _ = await generation_coordinator.submit(
            session_id, getattr(req, "request_id", None), "retry", runner
        )
        result = await job.wait()
    except SessionGenerationBusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except GenerationJobCancelledError as exc:
        raise HTTPException(499, str(exc)) from exc
    except GenerationEventError as exc:
        status_code = {
            "LLM_TIMEOUT": 504,
            "MODEL_BUSY": 409,
            "MODEL_RECOVERING": 503,
            "SESSION_BUSY": 409,
            "LLM_OOM": 507,
        }.get(exc.code, 400)
        raise HTTPException(status_code, str(exc)) from exc
    result.update({"job_id": job.job_id, "request_id": job.request_id})
    return StandardResponse(success=True, data=result)


@app.post("/api/sessions/{session_id}/retract")
async def retract_turn(session_id: str, turn: int):
    _ensure_session_writable(session_id)
    inst = session_mgr.retract_turn(session_id, turn)
    if not inst:
        raise HTTPException(404, "回合或会话不存在")
    snapshot = session_mgr.capture_snapshot(inst, operation="retract")
    return StandardResponse(success=True, data={
        "messages": inst.messages,
        "state": inst.build_state_snapshot().model_dump(),
        "snapshot_id": snapshot["snapshot_id"],
        "timeline_id": inst.timeline_id,
    })


@app.post("/api/sessions/{session_id}/rollback")
async def rollback_session(session_id: str, req: RollbackRequest):
    _ensure_session_writable(session_id)
    inst = session_mgr.rollback_to_snapshot(session_id, req.snapshot_id)
    if not inst:
        raise HTTPException(404, "快照或会话不存在")
    snapshot = session_mgr.capture_snapshot(inst, operation="rollback")
    return StandardResponse(success=True, data={
        "messages": inst.messages,
        "state": inst.build_state_snapshot().model_dump(),
        "snapshot_id": snapshot["snapshot_id"],
        "timeline_id": inst.timeline_id,
    })


# ====== 存档系统 ======
@app.get("/api/sessions/{session_id}/saves", response_model=list[SaveSlotSummary])
async def list_saves(session_id: str):
    inst = session_mgr.get_session(session_id)
    if not inst:
        raise HTTPException(404, "会话不存在")
    saves = session_mgr.list_saves(session_id)
    debug("存档", f"获取存档列表: session={session_id}, {len(saves)}个存档")
    return saves


@app.post("/api/sessions/{session_id}/saves")
async def create_save(session_id: str, req: CreateSaveRequest):
    inst = session_mgr.get_session(session_id)
    if not inst:
        raise HTTPException(404, "会话不存在")
    _ensure_session_writable(session_id)
    info("存档", f"创建存档: session={session_id}, name={req.name}")
    slot_id = session_mgr.create_save(session_id, req.name)
    if not slot_id:
        warning("存档", f"创建失败，会话不存在: {session_id}")
        raise HTTPException(404, "会话不存在")
    success("存档", f"✓ 存档已创建: slot={slot_id}")
    return StandardResponse(success=True, data={"slot_id": slot_id})


@app.post("/api/sessions/{session_id}/saves/load")
async def load_save(session_id: str, req: LoadSaveRequest):
    if not session_mgr.get_session(session_id):
        raise HTTPException(404, "会话不存在")
    _ensure_session_writable(session_id)
    info("存档", f"读取存档: session={session_id}, slot={req.slot_id}")
    inst = session_mgr.load_save(session_id, req.slot_id)
    if not inst:
        warning("存档", f"读取失败，存档不存在: slot={req.slot_id}")
        raise HTTPException(404, "存档不存在")
    success("存档", f"✓ 存档已读取，回合数: {inst.turn_count}")
    return StandardResponse(success=True, data={"state": inst.build_state_snapshot().model_dump()})


@app.delete("/api/sessions/{session_id}/saves/{slot_id}")
async def delete_save(session_id: str, slot_id: str):
    if not session_mgr.get_session(session_id):
        raise HTTPException(404, "会话不存在")
    _ensure_session_writable(session_id)
    info("存档", f"删除存档: session={session_id}, slot={slot_id}")
    ok = session_mgr.delete_save(session_id, slot_id)
    if not ok:
        raise HTTPException(404, "存档不存在")
    return StandardResponse(success=True)


@app.post("/api/sessions/{session_id}/branch")
async def branch_session(session_id: str, req: BranchRequest):
    if not session_mgr.get_session(session_id):
        raise HTTPException(404, "会话不存在")
    _ensure_session_writable(session_id)
    info("存档", f"创建分支: session={session_id}, from_slot={req.from_slot}, snapshot={req.snapshot_id}, name={req.name}")
    if req.snapshot_id:
        new_id = session_mgr.branch_from_snapshot(session_id, req.snapshot_id, req.name)
    elif req.from_slot:
        new_id = session_mgr.branch_session(session_id, req.from_slot, req.name)
    else:
        new_id = ""
    if not new_id:
        warning("存档", "分支创建失败")
        raise HTTPException(400, "分支创建失败")
    success("存档", f"✓ 分支已创建: {new_id}")
    return StandardResponse(success=True, data={"new_session_id": new_id})


# ====== 对话任务与流式传输 ======
def _resolve_chat_session(session_id: Optional[str]) -> str:
    sid = session_id
    if not sid:
        inst = session_mgr.get_active_session()
        if not inst:
            raise HTTPException(400, "没有活跃会话，请先创建会话")
        sid = inst.meta.session_id
    elif not session_mgr.get_session(sid):
        raise HTTPException(404, "会话不存在")
    return sid


async def _chat_runner(session_id: str, message: str, cancel_event, publish):
    result = {"response": "", "state": None, "messages": []}
    terminal_error = None
    async for event in orchestrator.process_message_stream(
        session_id, message, cancel_event=cancel_event, complete_response=False
    ):
        await publish(event)
        event_type = event.get("type")
        if event_type == "token":
            result["response"] += event.get("content", "")
        elif event_type == "done":
            result["response"] = event.get("full_response", result["response"])
        elif event_type == "state":
            result["state"] = event.get("data")
        elif event_type in {"error", "cancelled"}:
            terminal_error = event
    if terminal_error:
        code = terminal_error.get("code", "LLM_ERROR")
        message = terminal_error.get("message", "聊天处理失败")
        if code == "CANCELLED":
            raise GenerationJobCancelledError(message)
        raise GenerationEventError(code, message)
    inst = session_mgr.get_session(session_id)
    if not inst or not result["response"]:
        raise RuntimeError("聊天回合未完成")
    result["state"] = result["state"] or inst.build_state_snapshot().model_dump()
    result["messages"] = inst.messages[-2:]
    return result


def _chat_exception_event(exc: BaseException) -> dict:
    if isinstance(exc, GenerationJobCancelledError):
        return {"type": "cancelled", "code": "CANCELLED", "message": str(exc)}
    if isinstance(exc, GenerationEventError):
        return {"type": "error", "code": exc.code, "message": str(exc)}
    if isinstance(exc, llm_qwen.GenerationTimeoutError):
        return {"type": "error", "code": "LLM_TIMEOUT", "message": str(exc)}
    if isinstance(exc, llm_qwen.GenerationOOMError):
        return {"type": "error", "code": "LLM_OOM", "message": str(exc)}
    if isinstance(exc, llm_qwen.GenerationCancelledError):
        return {"type": "cancelled", "code": "CANCELLED", "message": str(exc)}
    return {"type": "error", "code": "LLM_ERROR", "message": str(exc) or "聊天处理失败"}


async def _submit_chat_job(sid: str, req: ChatRequest):
    return await generation_coordinator.submit(
        sid, req.request_id, "chat",
        lambda cancel_event, publish: _chat_runner(
            sid, req.message, cancel_event, publish
        ),
    )


async def _ndjson_chat_events(sid: str, req: ChatRequest):
    try:
        job, _ = await _submit_chat_job(sid, req)
    except Exception as exc:
        # Admission failures have no job identity or sequence because no
        # generation was created. They are still terminal transport events.
        event = _chat_exception_event(exc)
        event["terminal"] = True
        yield json.dumps(event, ensure_ascii=False) + "\n"
        return

    # GenerationJob owns generation lifetime and publishes exactly one
    # canonical terminal event. Cancelling/closing this consumer only
    # unsubscribes its queue; it never cancels the coordinator worker.
    async for event in job.stream():
        yield json.dumps(event, ensure_ascii=False) + "\n"


@app.post("/api/chat/{session_id}/cancel/{request_id}")
async def cancel_chat(session_id: str, request_id: str):
    status = generation_coordinator.cancel(session_id, request_id)
    return StandardResponse(success=True, data={"status": status})


@app.get("/api/generation/jobs/{job_id}")
async def generation_job_status(job_id: str):
    job = generation_coordinator.get_job(job_id)
    if job is None:
        raise HTTPException(404, "生成任务不存在")
    return StandardResponse(success=True, data=generation_coordinator.status(job))


@app.get("/api/generation/sessions/{session_id}")
async def session_generation_status(session_id: str):
    job = generation_coordinator.get_session_job(session_id)
    return StandardResponse(
        success=True,
        data=generation_coordinator.status(job) if job else None,
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_nonstream(req: ChatRequest, request: Request):
    sid = _resolve_chat_session(req.session_id)
    debug("编排器", f"[完整回复] 处理消息，长度: {len(req.message)}字")
    try:
        job, _ = await _submit_chat_job(sid, req)
        result = await job.wait()
    except SessionGenerationBusyError as exc:
        raise HTTPException(409, str(exc)) from exc
    except GenerationJobCancelledError as exc:
        raise HTTPException(499, str(exc)) from exc
    except GenerationEventError as exc:
        status_code = {"LLM_TIMEOUT": 504, "MODEL_BUSY": 409, "MODEL_RECOVERING": 503, "SESSION_BUSY": 409, "LLM_OOM": 507}.get(exc.code, 400)
        raise HTTPException(status_code, str(exc)) from exc
    except llm_qwen.GenerationTimeoutError as exc:
        raise HTTPException(504, str(exc)) from exc
    except llm_qwen.GenerationCancelledError as exc:
        raise HTTPException(499, str(exc)) from exc
    except llm_qwen.GenerationOOMError as exc:
        raise HTTPException(507, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(500, str(exc)) from exc
    inst = session_mgr.get_session(sid)
    return ChatResponse(
        response=result["response"],
        state=SessionStateSnapshot(**result["state"]),
        messages=inst.messages,
        job_id=job.job_id,
        request_id=job.request_id,
    )


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    sid = _resolve_chat_session(req.session_id)
    return StreamingResponse(
        _ndjson_chat_events(sid, req),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ====== WebSocket 流式对话 ======
@app.websocket("/ws/chat/{session_id}")
async def websocket_chat(websocket: WebSocket, session_id: str):
    await websocket.accept()
    debug("网络", f"WebSocket连接: {session_id}")
    
    inst = session_mgr.get_session(session_id)
    if not inst:
        warning("网络", f"WebSocket连接失败，会话不存在: {session_id}")
        await websocket.send_json({"type": "error", "message": "会话不存在"})
        await websocket.close()
        return

    session_mgr.switch_session(session_id)
    info("网络", f"WebSocket已连接: {session_id}")

    # 开场已经在创建事务中持久化；连接时重放同一条稳定消息。
    opening_message = inst.get_opening_message()
    if opening_message:
        debug("会话", f"发送开场消息: {opening_message['message_id']}")
        await websocket.send_json({
            "type": "opening",
            "content": opening_message.get("content", ""),
            "message": opening_message,
        })

    generation_task = None
    current_job = None
    receive_task = asyncio.create_task(websocket.receive_text())

    async def relay_generation(job):
        token_count = 0
        start_time = time.time()
        async for stream_event in job.stream():
            await websocket.send_json(stream_event)
            if stream_event["type"] == "token":
                token_count += len(stream_event.get("content", ""))
        await job.wait()
        debug("编排器", f"[流式] 生成完成，{token_count}字，耗时{time.time() - start_time:.1f}s")

    try:
        while True:
            if generation_task is None:
                data = await receive_task
            else:
                done, _ = await asyncio.wait(
                    {receive_task, generation_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if generation_task in done:
                    try:
                        generation_task.result()
                    except asyncio.CancelledError:
                        pass
                    except Exception as e:
                        error("编排器", f"处理消息出错: {e}", exc_info=True)
                        await websocket.send_json({
                            "type": "error",
                            "code": "CHAT_ERROR",
                            "message": "消息处理失败",
                        })
                    generation_task = None
                    current_job = None
                    if receive_task not in done:
                        continue
                data = receive_task.result()

            receive_task = asyncio.create_task(websocket.receive_text())
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                msg = {"type": "message", "content": data}

            message_type = msg.get("type")
            if message_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            if message_type == "stop":
                debug("编排器", "收到停止生成请求")
                request_id = msg.get("request_id")
                if request_id:
                    generation_coordinator.cancel(session_id, request_id)
                elif current_job:
                    generation_coordinator.cancel(session_id, current_job.request_id)
                continue
            if message_type != "message":
                await websocket.send_json({
                    "type": "error",
                    "code": "INVALID_MESSAGE_TYPE",
                    "message": "不支持的消息类型",
                })
                continue

            user_text = msg.get("content", "")
            if not user_text.strip():
                continue
            if generation_task is not None:
                await websocket.send_json({
                    "type": "error",
                    "code": "BUSY",
                    "message": "上一条消息仍在生成",
                })
                continue

            debug("编排器", f"[流式] 收到用户消息，长度: {len(user_text)}字")

            async def runner(cancel_event, publish):
                async for stream_event in orchestrator.process_message_stream(
                        session_id, user_text, cancel_event=cancel_event):
                    await publish(stream_event)

            try:
                current_job, _ = await generation_coordinator.submit(
                    session_id, msg.get("request_id"), "chat", runner
                )
            except SessionGenerationBusyError:
                await websocket.send_json({
                    "type": "error",
                    "code": "SESSION_BUSY",
                    "message": "上一条消息仍在生成",
                })
                continue
            generation_task = asyncio.create_task(relay_generation(current_job))
    except WebSocketDisconnect:
        info("网络", f"WebSocket断开: {session_id}")
    except Exception as e:
        error("网络", f"WebSocket异常: {e}", exc_info=True)
        try:
            await websocket.send_json({
                "type": "error",
                "code": "WEBSOCKET_ERROR",
                "message": "WebSocket连接异常",
            })
        except Exception:
            pass
    finally:
        # 传输断开只停止转发；协调器中的生成任务继续在后台运行。
        pending = []
        for task in (receive_task, generation_task):
            if task is not None and not task.done():
                task.cancel()
            if task is not None:
                pending.append(task)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


# ====== 日志查看接口（轻量可视化）====== 
import re as _re


@app.get("/api/logs")
async def list_logs(limit: int = 300, file: str = None, level: str = None,
                    module: str = None, q: str = None):
    """读取最新日志并返回结构化记录，供前端日志查看器实时刷新。"""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(base, "logs")
    if file:
        path = os.path.join(log_dir, file)
        if not os.path.exists(path):
            return StandardResponse(success=False, data={"error": "日志文件不存在", "file": file})
    else:
        if os.path.isdir(log_dir):
            files = sorted(f for f in os.listdir(log_dir)
                           if f.startswith("heartchat_") and f.endswith(".log"))
        else:
            files = []
        path = os.path.join(log_dir, files[-1]) if files else None
    if not path or not os.path.exists(path):
        return StandardResponse(success=True, data={"file": "", "mtime": 0, "total": 0, "lines": [], "modules": []})
    # 只读文件末尾一段（多行块 + 过滤），避免全量加载
    tail_bytes = 2_000_000
    with open(path, "rb") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        fh.seek(max(0, size - tail_bytes))
        data = fh.read().decode("utf-8", errors="replace")
    records = []
    cur = None
    pat = _re.compile(r"^\[([\d:.]+)\]\[ ?([\d.]+)s\]\[(\w+)\]\[([^\]]+)\] ?(.*)$")
    for line in data.splitlines():
        m = pat.match(line)
        if m:
            cur = {"ts": m.group(1), "uptime": m.group(2), "level": m.group(3),
                   "module": m.group(4), "msg": m.group(5), "detail": []}
            records.append(cur)
        elif cur is not None and line.strip():
            cur["detail"].append(line.rstrip())
    all_modules = sorted({r["module"] for r in records})
    if level:
        records = [r for r in records if r["level"] == level.upper()]
    if module:
        records = [r for r in records if r["module"] == module]
    if q:
        ql = q.lower()
        records = [r for r in records if ql in r["msg"].lower()
                   or any(ql in d.lower() for d in r["detail"])]
    total = len(records)
    lines = records[-max(1, int(limit)):]
    return StandardResponse(success=True, data={
        "file": os.path.basename(path),
        "mtime": os.path.getmtime(path),
        "total": total,
        "lines": lines,
        "modules": all_modules,
    })


# ====== 启动入口 ======
def start():
    import uvicorn
    info("服务器", "通过uvicorn启动服务...")
    uvicorn.run("app.server:app", host=SERVER_HOST, port=SERVER_PORT,
                reload=False, log_level="warning")  # 用warning级别避免uvicorn自己的日志重复


if __name__ == "__main__":
    start()
