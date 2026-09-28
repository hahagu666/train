"""
终端直聊测试入口（独立脚本，不修改原项目任何文件）

用途：在终端里直接和固定的"妹妹"角色对话，测试
  - 对话生成
  - 长期记忆 / 知识库
  - 信任、亲密度、接受度、生理阶段等各项状态

启动：
    python terminal_chat.py            # 新开一局（自动随机开场剧情）
    python terminal_chat.py --resume   # 恢复最近一次妹妹会话

会话数据仍写入 data/sessions/，与 Web 端共用持久化层，可以随时互相续聊。

终端内命令：
    /help      命令列表
    /state     显示完整状态快照
    /memory    显示角色当前记忆
    /history   显示最近对话
    /new       新开一局
    /sessions  列出所有妹妹会话
    /open <id> 切换到指定会话
    /quit      退出
"""
import asyncio
import functools
import os
import sys
import threading
import traceback
from datetime import datetime

# Windows CMD 默认 GBK，强制 UTF-8 避免中文乱码
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

from app.character_manager import CharacterManager
from app.user_profile import UserProfileManager
from app.session_manager import SessionManager
from app.knowledge_service import KnowledgeService
from app.chat_orchestrator import ChatOrchestrator
from app.logger import Logger, trace as log_trace
import app.logger as _logger_mod
import llm_qwen
import llm_small

# ===== 专用trace日志：每次运行一个独立文件，记录全部运行过程 =====
TRACE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "terminal")


class _TraceLogger(Logger):
    """继承 Heart Chat 日志器，但把每次终端运行写入独立的 trace 文件。

    这样解析、身体引擎、Prompt组装、大模型、后处理、记忆、存档等
    所有过程日志 + 完整Prompt/回复都按顺序收集在同一个文件里。
    """

    def _open_log_file(self):
        os.makedirs(self.log_dir, exist_ok=True)
        filename = f"trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        path = os.path.join(self.log_dir, filename)
        self._log_file = open(path, "a", encoding="utf-8", buffering=1)
        self._write_file_raw("=" * 80)
        self._write_file_raw(f"终端对话trace日志 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self._write_file_raw("=" * 80)


def _setup_trace_logger() -> str:
    """把全局日志器换成trace版本，返回日志文件路径。"""
    Logger._instance = None
    trace_logger = _TraceLogger(log_dir=TRACE_DIR)
    _logger_mod._logger_instance = trace_logger
    return trace_logger._log_file.name

# ===== 固定设定：妹妹 林小雨（imouto）=====
CHAR_ID = "imouto"
USER_PROFILE = {
    "name": "哥哥",
    "age": 21,
    "gender": "男",
    "relation": "brother",
    "preferred_address": "哥哥",
    "personality_description": "平时嘴硬心软，很宠妹妹",
    "additional_context": "和妹妹一起住，父母经常不在家",
}

HELP_TEXT = """命令列表：
  /help      显示本帮助
  /state     显示完整状态快照（信任/亲密/接受度/生理阶段等）
  /memory    显示角色当前记忆内容
  /history   显示最近10条对话
  /new       新开一局（随机开场剧情）
  /sessions  列出所有妹妹会话
  /open <id> 切换到指定会话继续聊
  /quit      退出
直接输入文字即可对话，动作请用全角括号，例如：（轻轻抱住她）"""

# 情绪英文键 -> 中文（与 Web 端 emotionLabels 保持一致）
EMOTION_LABELS = {
    "calm": "平静", "happy": "开心", "happiness": "开心", "joy": "喜悦",
    "excited": "兴奋", "excitement": "兴奋", "shy": "害羞", "shyness": "害羞",
    "shame": "羞耻", "embarrassment": "尴尬", "nervous": "紧张",
    "anxious": "不安", "anxiety": "不安", "sad": "难过", "hurt": "委屈",
    "angry": "生气", "frustration": "烦闷", "fear": "害怕", "shock": "震惊",
    "surprised": "惊讶", "surprise": "惊讶", "love": "心动", "pleasure": "愉悦",
    "jealousy": "吃醋", "anticipation": "期待", "satisfaction": "满足",
    "overwhelm": "不知所措", "loss_of_control": "失控",
    "inevitability": "难以抗拒", "pain": "疼痛", "sleepy": "困倦",
    "longing": "渴望", "playfulness": "俏皮", "trust": "信任", "neutral": "平静",
}


def _localize_emotion(value) -> str:
    """把 'hurt/frustration+shame' 这类英文情绪串转成中文。"""
    if not value:
        return "平静"
    text = str(value)
    segments = []
    for part in text.split("/"):
        subs = [EMOTION_LABELS.get(sub.strip().lower(), sub.strip())
                for sub in part.split("+")]
        segments.append("+".join(subs))
    return "/".join(segments)


def _make_async_small_llm():
    """与 server.py 相同的包装：同步小模型放到线程池，避免阻塞事件循环。"""
    loop = asyncio.get_event_loop()

    async def _async_generate(prompt: str, max_new_tokens: int = 200,
                              temperature: float = 0.3, **kw) -> str:
        fn = functools.partial(llm_small.generate, prompt,
                               max_new_tokens=max_new_tokens,
                               temperature=temperature, **kw)
        return await loop.run_in_executor(None, fn)

    return _async_generate


def _print_divider(title: str = ""):
    if title:
        print(f"\n──── {title} ────")
    else:
        print("\n" + "─" * 40)


def _format_snapshot(snap) -> str:
    """把状态快照（对象或dict）整理成易读的几行。"""
    if isinstance(snap, dict):
        def g(key, default=None):
            return snap.get(key, default)
    else:
        def g(key, default=None):
            return getattr(snap, key, default)

    phase_names = {
        "baseline": "基线（未唤起）", "excitement": "兴奋期",
        "plateau": "平台期", "orgasm": "高潮", "resolution": "消退期",
    }
    stage = g("stage", "?")
    scenario_stage = g("scenario_stage")
    if scenario_stage:
        stage = f"{stage}  剧情:{scenario_stage}"
    lines = [
        f"关系   阶段:{stage}  信任:{float(g('trust', 0)):.2f}  "
        f"亲密:{float(g('closeness', 0)):.2f}  接受度:{float(g('acceptance', 0)):.2f}",
        f"生理   阶段:{phase_names.get(g('phase', ''), g('phase', '?'))}  "
        f"唤起:{float(g('arousal', 0)):.2f}  心率:{g('heart_rate', '-')}  "
        f"体力:{float(g('stamina', 0)):.2f}  疲劳:{float(g('fatigue', 0)):.2f}",
        f"情绪   {_localize_emotion(g('emotion'))}",
        f"场景   {g('privacy', '-')}（隐私度 {float(g('privacy_level', 0)):.2f}）  "
        f"危险:{float(g('danger_level', 0)):.2f}  被发现风险:{float(g('detection_risk', 0)):.2f}",
    ]
    if g("is_interrupted", False):
        lines.append("⚠ 本回合存在中断事件")
    return "\n".join(lines)


def _print_memory(inst):
    mem = getattr(inst.char_state, "memory", None)
    if mem is None:
        print("（当前没有记忆数据）")
        return
    _print_divider("记忆")
    if isinstance(mem, dict):
        for key, value in list(mem.items())[:20]:
            print(f"{key}: {value}")
        return
    # MemorySystem 的真实字段
    summary_pairs = [
        ("正向记忆总量", "total_positive_bond"),
        ("未愈合伤害", "total_hurt_unhealed"),
        ("创伤敏感度", "trauma_sensitivity"),
        ("近期情绪倾向", "recent_valence"),
    ]
    for label, attr in summary_pairs:
        value = getattr(mem, attr, None)
        if value is not None:
            print(f"{label}: {value}")
    positives = getattr(mem, "positive_memories", None)
    if positives:
        print("[正向记忆]")
        for item in list(positives)[-10:]:
            print(f"  - {item}")
    hurts = getattr(mem, "hurt_episodes", None)
    if hurts:
        print("[伤害事件]")
        for item in list(hurts)[-10:]:
            print(f"  - {item}")
    if not any(getattr(mem, attr, None) for _, attr in summary_pairs) \
            and not positives and not hurts:
        print("（暂无值得记录的记忆）")


class TerminalChat:
    def __init__(self):
        small_llm = _make_async_small_llm() if llm_small.is_available() else None
        self.char_mgr = CharacterManager(small_llm_fn=small_llm)
        self.user_mgr = UserProfileManager(small_llm_fn=small_llm)
        self.knowledge_svc = KnowledgeService()
        self.session_mgr = SessionManager(
            character_manager=self.char_mgr,
            knowledge_service=self.knowledge_svc,
        )
        self.orchestrator = ChatOrchestrator(
            char_mgr=self.char_mgr,
            session_mgr=self.session_mgr,
            user_mgr=self.user_mgr,
            knowledge_svc=self.knowledge_svc,
            small_llm_fn=small_llm,
        )
        self._model_ready = None  # run() 中后台加载主模型的 future
        self.session_id = None
        self.char_name = "林小雨"

    # ===== 会话管理 =====
    def new_session(self):
        inst = self.session_mgr.create_session(
            char_id=CHAR_ID, name="终端测试", user_profile=USER_PROFILE,
        )
        self.session_id = inst.meta.session_id
        self.char_name = getattr(
            self.char_mgr.get_character(CHAR_ID), "name", "林小雨")
        opening = inst.get_opening_message()
        _print_divider(f"新会话 {self.session_id}")
        if opening and opening.get("content"):
            print(f"{self.char_name}: {opening['content']}")
        else:
            print("（没有开场剧情，直接开始对话即可）")
        return inst

    def resume_latest(self) -> bool:
        sessions = self.session_mgr.list_sessions(char_id=CHAR_ID)
        if not sessions:
            return False
        latest = sessions[0]  # list_sessions 已按 last_active_at 倒序
        inst = self.session_mgr.switch_session(latest.session_id)
        if not inst:
            return False
        self.session_id = latest.session_id
        self.char_name = getattr(
            self.char_mgr.get_character(CHAR_ID), "name", "林小雨")
        _print_divider(f"恢复会话 {self.session_id}（{latest.session_name}，"
                       f"回合数 {latest.total_interactions}）")
        for message in inst.messages[-6:]:
            role = "你" if message.get("role") == "user" else self.char_name
            print(f"{role}: {message.get('content', '')}")
        return True

    def open_session(self, sid: str):
        inst = self.session_mgr.switch_session(sid.strip())
        if not inst:
            print(f"会话不存在: {sid}")
            return
        self.session_id = sid
        _print_divider(f"已切换到 {sid}")
        for message in inst.messages[-6:]:
            role = "你" if message.get("role") == "user" else self.char_name
            print(f"{role}: {message.get('content', '')}")

    # ===== 单轮对话 =====
    async def chat_turn(self, user_input: str):
        # 模型仍在后台加载时先等它就绪，否则会被"模型忙"拒绝
        ready = getattr(self, "_model_ready", None)
        if ready is not None and not ready.done():
            print("（等待主模型加载完成...）", end="", flush=True)
            try:
                await asyncio.wait_for(asyncio.shield(ready), timeout=300)
            except Exception as e:
                print(f"\n✗ 主模型加载失败: {e}")
                log_trace("终端", f"主模型加载失败:\n{traceback.format_exc()}")
                return
            print(" 完成")
        cancel_event = threading.Event()
        print(f"\n{self.char_name}: ", end="", flush=True)
        full_response = ""
        try:
            async for event in self.orchestrator.process_message_stream(
                    self.session_id, user_input, cancel_event=cancel_event):
                etype = event.get("type")
                if etype == "token":
                    token = event.get("content", "")
                    full_response += token
                    print(token, end="", flush=True)
                elif etype == "blocked":
                    reason = event.get("reason", "")
                    print(f"\n⚠ 动作被阻止 [{event.get('code')}] "
                          f"动作={event.get('action', '')}"
                          f"{'：' + reason if reason else ''}")
                    log_trace("终端", f"动作被阻止: {event}")
                elif etype == "state":
                    print()  # 结束回复行
                    print(_format_snapshot(event.get("data")))
                    log_trace("终端", "状态摘要：\n" + _format_snapshot(event.get("data")))
                elif etype == "error":
                    print(f"\n✗ 出错: {event.get('message', event)}")
                    log_trace("终端", f"错误事件: {event}")
                elif etype == "memory_formed":
                    print(f"\n💭 她记住了: {event.get('summary', '')}")
                # action_start / generating 等过程事件在终端静默
        except KeyboardInterrupt:
            cancel_event.set()
            print("\n（已取消本轮生成）")
        except Exception as e:
            print(f"\n✗ 对话处理异常: {e}")
            log_trace("终端", f"对话处理异常:\n{traceback.format_exc()}")
        if full_response:
            log_trace("终端", f"本轮完整回复：{full_response}")
        print()
        log_trace("终端", "──── 等待下一次用户输入 ────")

    # ===== REPL =====
    async def run(self, resume: bool):
        print("=" * 60)
        print("Heart Chat 终端测试（固定角色：妹妹 林小雨）")
        print("=" * 60)

        # 后台预热主模型，第一条消息不用干等
        loop = asyncio.get_event_loop()
        self._model_ready = loop.run_in_executor(None, llm_qwen.load_model)
        print("（主模型正在后台加载，首次回复可能稍慢）")

        if not (resume and self.resume_latest()):
            self.new_session()

        print(HELP_TEXT)

        while True:
            try:
                user_input = input("\n你> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break
            if not user_input:
                continue

            # 每一条输入（含命令）都进入 trace 日志
            log_trace("终端", f"──── 用户输入：{user_input} ────")

            if user_input in ("/quit", "/exit", "/q"):
                print("再见。")
                break
            if user_input.startswith("/"):
                # 命令处理整体兜底：任何异常只提示，绝不让 REPL 退出
                try:
                    handled = self._dispatch_command(user_input)
                except Exception as e:
                    print(f"✗ 命令处理异常: {e}")
                    log_trace("终端", f"命令处理异常:\n{traceback.format_exc()}")
                    handled = True
                if handled:
                    continue
                print(f"未知命令: {user_input}（输入 /help 查看帮助）")
                continue

            await self.chat_turn(user_input)

    def _dispatch_command(self, cmd: str) -> bool:
        """处理 / 开头的命令；返回 True 表示已处理，False 表示未知命令。"""
        if cmd == "/help":
            print(HELP_TEXT)
            return True
        if cmd == "/new":
            self.new_session()
            return True
        if cmd == "/state":
            inst = self.session_mgr.get_session(self.session_id)
            if inst:
                print(_format_snapshot(inst.build_state_snapshot()))
            return True
        if cmd == "/memory":
            inst = self.session_mgr.get_session(self.session_id)
            if inst:
                _print_memory(inst)
            return True
        if cmd == "/history":
            inst = self.session_mgr.get_session(self.session_id)
            if inst:
                _print_divider("最近对话")
                for message in inst.messages[-10:]:
                    role = "你" if message.get("role") == "user" else self.char_name
                    print(f"{role}: {message.get('content', '')}")
            return True
        if cmd == "/sessions":
            _print_divider("妹妹会话列表")
            for meta in self.session_mgr.list_sessions(char_id=CHAR_ID):
                mark = " ←当前" if meta.session_id == self.session_id else ""
                active = meta.last_active_at
                active_str = (active.strftime("%m-%d %H:%M")
                              if hasattr(active, "strftime") else str(active))
                print(f"{meta.session_id}  回合{meta.total_interactions:>4}  "
                      f"{active_str}{mark}")
            return True
        if cmd.startswith("/open "):
            self.open_session(cmd[6:])
            return True
        return False


def _force_utf8_stdio():
    """管道输入/输出统一按 UTF-8 解码。

    Windows 下 Python 对管道默认用 GBK+surrogateescape，piped UTF-8 中文会变成
    带孤立代理字符的乱码：轻则日志写不进文件，重则模型 tokenizer 直接报错
    （TextEncodeInput must be ...）。交互式控制台走宽字符 API，不受影响。
    """
    try:
        if not sys.stdin.isatty():
            sys.stdin.reconfigure(encoding="utf-8", errors="replace")
        if not sys.stdout.isatty():
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if not sys.stderr.isatty():
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


def main():
    _force_utf8_stdio()
    resume = "--resume" in sys.argv
    # 独立 trace 日志：本次运行所有过程日志写入 logs/terminal/trace_*.log
    log_path = _setup_trace_logger()
    print(f"日志文件： {log_path}")
    log_trace("终端", f"程序启动 resume={resume}")
    app = TerminalChat()
    try:
        asyncio.run(app.run(resume))
    except KeyboardInterrupt:
        print("\n再见。")
    finally:
        # 兜底：无论正常/异常退出都释放模型 worker
        try:
            llm_qwen.unload_model()
        except Exception:
            pass
        log_trace("终端", "程序退出")


if __name__ == "__main__":
    main()
