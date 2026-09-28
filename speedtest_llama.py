# -*- coding: utf-8 -*-
"""llama.cpp (GGUF CUDA) 速度实测：应用真实 prompt 形态 + 长生成稳态 tok/s。"""
import time
import llm_llama


def realistic_prompt() -> str:
    sys_rules = (
        "# 安全与输出规则\n台词直接写，动作用全角括号（ ）。2-5句话。\n"
        "# 角色身份资料\n<DATA:CHARACTER_IDENTITY>\n"
        '{"name":"小月","age":19,"character_description":"温柔体贴的妹妹","backstory":"相依为命",'
        '"personality":{"kind":0.9,"shy":0.6},"speech_style":{"first_person":"我"}}\n'
        "</DATA:CHARACTER_IDENTITY>\n"
        "# 用户身份资料\n<DATA:USER_IDENTITY>\n{\"name\":\"哥哥\",\"preferred_address\":\"哥哥\"}\n</DATA:USER_IDENTITY>"
    )
    state = (
        "## 已提交场景与仿真状态\n<DATA:COMMITTED_STATE>\n"
        '{"scene":"现在是晚上21:30，你们在客厅沙发上。","relationship":{"trust":0.62,"closeness":0.55,'
        '"stage":"C"},"physiology":{"heart_rate":78,"arousal":0.2,"emotion":"平静"},'
        '"sensations":"她调整了一下坐姿。"}\n</DATA:COMMITTED_STATE>'
    )
    history = (
        "## 最近完整回合\n<DATA:RECENT_COMPLETE_TURNS>\n"
        '[{"turn":1,"messages":[{"speaker":"对方（哥哥）","text":"今天上班累吗？"},'
        '{"speaker":"你（小月）","text":"还好，就是有点想你了（靠在你肩上）。"}]}]\n</DATA:RECENT_COMPLETE_TURNS>'
    )
    return (
        sys_rules + "\n" + state + "\n" + history
        + "\n## 当前输入\n<DATA:CURRENT_INPUT>\n{\"text\":\"过来，让我抱抱你。\"}\n</DATA:CURRENT_INPUT>\n"
        + "\n现在轮到你说话：以角色身份直接回应。"
    )


def run(label, prompt, max_new_tokens, temperature):
    t0 = time.time()
    content = llm_llama.generate(prompt, system_prompt="", max_new_tokens=max_new_tokens, temperature=temperature)
    elapsed = time.time() - t0
    s = llm_llama._last_generation
    print(f"\n[{label}] 生成 {len(content)}字, 总耗时 {elapsed:.2f}s")
    print(f"  prompt_tokens={s['prompt_tokens']} output_tokens={s['output_tokens']} "
          f"tok/s(总)={s['tokens_per_second']} stop={s['stop_reason']}")
    print(f"  输出: {content[:60]}...")


if __name__ == "__main__":
    print("is_loaded =", llm_llama.is_loaded())
    if not llm_llama.is_loaded():
        print("调用 load_model 启动...")
        print("load_model =", llm_llama.load_model())
    # 1) 应用真实形态：长系统+状态 prompt，短生成
    run("应用真实形态(长prompt,短生成)", realistic_prompt(), 100, 0.8)
    # 2) 短 prompt + 长生成 => 稳态 decode tok/s
    run("稳态decode(短prompt,长生成)", "她轻轻握住你的手，在你耳边说：", 200, 0.8)
