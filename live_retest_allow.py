# -*- coding: utf-8 -*-
"""复测：恢复上午的会话（acceptance=0.55, response_level=allow），
验证收紧后的 allow 档规则下，亲密动作的回应是否从"推脱"变为"害羞配合"。"""
import asyncio

import llm_qwen
import llm_small
from terminal_chat import TerminalChat


async def main():
    tc = TerminalChat()
    if not llm_qwen.load_model():
        print("!! 主模型加载失败")
        return
    # 恢复最近会话（92e96f4a0ab8 是上午的测试会话）
    if not tc.resume_latest():
        print("!! 没有可恢复的会话")
        return
    w = tc.session_mgr.get_session(tc.session_id).world
    w.scenario_people_override = False
    w.people_present = {
        "user": {"location": "current", "activity": None},
        "sister": {"location": "current", "activity": None},
    }
    w.weekday = "Tuesday"
    w.game_time.set_time(10, 0)
    w._update_parents_presence()
    w.set_location("bedroom")
    w.door_locked = True
    print(f"场景: 周二上午10点 卧室锁门 parents_present={w.parents_present}")

    await tc.chat_turn("（把她轻轻拉进怀里，吻住她的唇）")
    await tc.chat_turn("（吻得深一点，手轻轻抚过她的后背）")

    inst = tc.session_mgr.get_session(tc.session_id)
    m = inst.char_state.mind
    print(f"\n最终: acceptance={m.acceptance_level:.2f} "
          f"resistance={m.active_resistance_will:.2f} "
          f"refusal={m.refusal_sincerity:.2f}")


if __name__ == "__main__":
    asyncio.run(main())
