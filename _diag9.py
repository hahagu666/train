# -*- coding: utf-8 -*-
import urllib.request, json, time

def post(url, payload, timeout=150):
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"_http_error": e.code, "_body": e.read().decode()[:200]}
    except Exception as e:
        return {"_err": str(e)[:150]}

r = post("http://127.0.0.1:8000/api/sessions", {
    "character_id": "imouto", "name": "完整质量测试",
    "scenario_id": "D-I01",
    "initial_trust": 0.9, "initial_closeness": 0.85, "privacy": "私密",
})
if "_http_error" in r:
    print("建会话失败:", r); raise SystemExit(1)
sid = r["data"]["session_id"]
print("会话:", sid)

def chat(msg):
    t0 = time.time()
    d = post("http://127.0.0.1:8000/api/chat", {"message": msg, "session_id": sid})
    dt = time.time() - t0
    if "_http_error" in d or "_err" in d:
        print(f"  [失败 {d.get('_http_error', d.get('_err',''))}] {msg[:20]}"); return
    st = d.get("state", {})
    print(f"\n>>> {msg}")
    print(f"    妹妹({dt:.1f}s): {d.get('response','')}")
    print(f"    状态: stage={st.get('stage')} acc={st.get('acceptance')} arousal={st.get('arousal')}")

chat("过来。")
chat("（我搂住你，摸摸你的头）我的软软长大了。")
chat("我低头轻轻吻住你。你知道自己在说什么吗？")
chat("（我亲亲你的耳朵，摸摸你的背）怕就停下来。")
chat("你都湿了。（我摸到你内裤湿了，把睡裙推上去）")
chat("（我含着你的乳头，手伸进你内裤摸你的小穴）你的乳头好硬。")
chat("我进去了，疼就告诉我。（肉棒抵在你穴口）")
chat("（我慢慢顶进去，整根没入）嘶……")
chat("（我慢慢抽送，一下一下顶到最深处）乖，忍一下。")
chat("（我加快抽送，顶到你最里面）要到了吗？")
