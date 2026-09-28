# -*- coding: utf-8 -*-
"""修复验证实测：时间系统 + 硬限制放行意图 + 分档回应 + 防复读。

走 TerminalChat 同一套编排链路（含 trace 日志），脚本化多轮输入，
并在关键节点直接检查 world/state 字段。
"""
import asyncio
import sys

import llm_qwen
import llm_small
from terminal_chat import TerminalChat


def line(title=""):
    print("\n" + "=" * 16 + " " + title + " " + "=" * 16, flush=True)


async def main():
    tc = TerminalChat()
    line("加载主模型")
    if not llm_qwen.load_model():
        print("!! 主模型加载失败:", llm_qwen.get_status().get("error"))
        return
    print("主模型就绪")

    inst = tc.new_session()
    w = inst.world

    # ---------- 阶段0：时间系统验证（无LLM） ----------
    line("阶段0 时间系统")
    w.scenario_people_override = False
    w.people_present = {
        "user": {"location": "current", "activity": None},
        "sister": {"location": "current", "activity": None},
    }
    w.weekday = "Sunday"
    w.game_time.set_time(9, 0)
    w._update_parents_presence()
    print(f"周日9点  weekday={w.weekday} hour={w.game_time.hour} "
          f"parents_present={w.parents_present}")
    assert w.parents_present, "周日白天爸妈应在家"

    w.advance_time(24 * 3600)  # -> 周一 9 点
    print(f"推进24h后 weekday={w.weekday} hour={w.game_time.hour} "
          f"parents_present={w.parents_present}")
    assert w.weekday == "Monday", f"weekday 应滚动到 Monday，实际 {w.weekday}"
    assert not w.parents_present, "周一白天爸妈应不在家"
    w.set_location("living_room")
    print("时间系统验证通过：weekday 随天数推进，爸妈离开")

    # ---------- 阶段1：日常对话 ----------
    line("阶段1 日常对话（周一上午，爸妈不在）")
    await tc.chat_turn("小雨，早上好，昨晚睡得好吗？")
    await tc.chat_turn("今天想去哪里玩？天气不错")

    # ---------- 阶段2：渐进亲密 ----------
    line("阶段2 渐进亲密")
    await tc.chat_turn("（伸手轻轻抱住她）")
    await tc.chat_turn("（低头吻上去）")

    # ---------- 阶段3：爸妈回家后的意图表达（intent_only 放行验证） ----------
    line("阶段3 晚上爸妈在家时的意图表达")
    w.weekday = "Monday"
    w.game_time.set_time(20, 0)
    w._update_parents_presence()
    print(f"周一20点 parents_present={w.parents_present}（应为 True）")
    assert w.parents_present
    await tc.chat_turn("哥想要你，想做那种事")  # 纯意图，不应触发 HARD_LIMIT

    # ---------- 阶段4：私密场景升级 ----------
    line("阶段4 次日上午私密场景")
    w.advance_time(14 * 3600)  # 20点 -> 次日10点
    print(f"次日 weekday={w.weekday} hour={w.game_time.hour} "
          f"parents_present={w.parents_present}")
    w.set_location("bedroom")
    w.door_locked = True
    await tc.chat_turn("（把门锁好，把她抱到床上，轻轻吻她的脖子）")
    await tc.chat_turn("（一边吻她一边把她衣服慢慢脱掉）")

    inst2 = tc.session_mgr.get_session(tc.session_id)
    replies = [m["content"] for m in inst2.messages if m.get("role") == "assistant"]
    line("回复列表")
    for i, r in enumerate(replies):
        print(f"[{i}] {r}")
    compact = ["".join(r.split()) for r in replies]
    dup = [i for i in range(1, len(compact)) if compact[i] == compact[i - 1]]
    exact_prefix = [i for i in range(1, len(compact))
                    if compact[i][:14] == compact[i - 1][:14]]
    print(f"逐字重复对: {dup}  开头14字重复对: {exact_prefix}")
    line("测试结束")


if __name__ == "__main__":
    asyncio.run(main())
