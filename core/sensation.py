"""
Layer 1 - Sensation Generator 感受生成器
根据Layer 0的数值状态，多维度概率采样生成具体的感受标签
供LLM组装成自然语言回应，确保每次回应都有身体细节且不重复

采样维度：
1. touch_sensation: 接触质感
2. temperature: 温度感受
3. physical_response: 身体生理反应（心跳/呼吸/颤抖）
4. local_pleasure: 局部快感（部位+感觉类型）
5. emotional_feeling: 情绪感受
6. action_urge: 动作冲动（想躲开/想贴近/想抱紧）
7. vocalization: 声音/呼吸反应
8. cognitive_snippet: 脑海中闪过的念头碎片
9. body_awareness: 身体自我察觉（意识到自己湿了/脸红了等）
"""
import random
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


# ============ 词库 ============

TOUCH_SENSATIONS = {
    # 通用
    "soft": ["柔软", "温热", "细腻", "光滑", "暖暖的", "软乎乎"],
    "rough": ["粗糙", "有点扎", "硬硬的"],
    "electric": ["触电般", "一阵酥麻", "像电流窜过", "麻酥酥"],
    "tickling": ["痒痒的", "有点痒", "轻搔"],
    "pressure": ["被按紧", "压着", "抵着", "顶着"],
    "rhythmic": ["有节奏地", "一下一下", "规律地"],
    "teasing": ["若有似无", "擦过", "蹭过", "轻轻掠过", "故意似的"],
    "penetrating": ["撑开", "填满", "胀胀的", "被撑开", "顶到最深处"],
    "friction": ["摩擦着", "蹭着", "磨着"],
    "wet_slide": ["滑溜溜", "湿漉漉地", "润滑地滑过"],
    # 部位特化
    "nipple": ["被含住", "被吸吮", "被舔弄", "被夹住", "硬硬地立着被摩擦"],
    "clitoris": ["被拨动", "被揉弄", "麻酥地跳", "一阵阵快感", "敏感地颤抖"],
    "inside": ["内壁被摩擦", "深处被顶到", "被搅动", "被撑开塞满", "收缩着绞紧"],
    "throbbing": ["突突地跳着", "一阵阵酥麻", "跳动着发出快感"],
}

TEMPERATURE_SENSATIONS = {
    "hot": ["发烫", "热热的", "火烧火燎", "灼热"],
    "warm": ["暖暖的", "温热"],
    "cold_shock": ["一阵凉意", "凉得一颤", "打了个寒颤"],
    "flushing": ["脸颊发烫", "脸烧起来", "胸口发烫", "耳朵烫得厉害"],
}

PHYSICAL_RESPONSES = {
    "heart": {
        "fast": ["心跳得很快", "心脏怦怦跳", "心跳快得像要撞出来"],
        "pounding": ["心怦怦直跳", "心脏咚咚地撞着胸口"],
        "racing": ["心跳快得喘不过气"],
    },
    "breathing": {
        "fast": ["呼吸变急促", "呼吸乱了", "喘息着", "气喘吁吁"],
        "shallow": ["呼吸很浅", "浅浅地喘着", "小口喘气"],
        "holding": ["屏住了呼吸", "倒抽一口冷气", "呼吸一窒"],
        "heavy": ["呼吸变得沉重", "粗重地喘息", "喘着粗气"],
        "gasping": ["倒吸凉气", " gasped", "啊地倒抽一口气"],
    },
    "muscle": {
        "tremble": ["身体微微发颤", "身子发软", "腿在抖", "手指在抖", "止不住地颤"],
        "tense": ["身体绷紧了", "背弓起来", "脚趾蜷起来", "手指抓紧床单"],
        "weak": ["浑身发软", "力气像被抽走", "腿软了", "站不住"],
        "spasm": ["一阵痉挛", "不受控制地缩紧", "抽搐着"],
        "arch": ["腰忍不住弓起来", "背弓成好看的弧度"],
    },
    "skin": {
        "flush": ["脸颊和胸口泛着粉", "皮肤泛起粉红", "粉晕从脸颊蔓延到胸口"],
        "goosebump": ["起了一层鸡皮疙瘩"],
        "sweat": ["出了一层薄汗", "微微出汗", "皮肤黏黏的"],
    },
}

LOCAL_PLEASURE = {
    "building": ["快感慢慢累积着", "一股热流慢慢聚集", "麻酥感慢慢扩散"],
    "spreading": ["快感从小腹扩散开", "那股热流蔓延开来", "酥麻感传遍全身"],
    "sharp": ["一阵尖锐的快感", "猛地窜过一阵快感", "强烈的快感让人一僵"],
    "throbbing": ["那里突突地跳着", "一阵阵酥麻袭来", "跳动着发出快感"],
    "overwhelming": ["快感像潮水一样涌来", "快感到让人无法思考", "快感淹没了意识"],
    "plateau": ["快感累积着快要到顶点了", "悬在临界点不上不下", "快要到了快要到了"],
    "waves": ["一波又一波的快感", "快感一浪接一浪"],
}

EMOTIONAL_FEELINGS = {
    "shy": ["害羞", "不好意思", "脸烧得厉害", "想躲起来"],
    "want_more": ["想要更多", "还想要", "不满足", "贪心的"],
    "conflicted": ["心里乱糟糟的", "不知道该怎么办", "一边羞耻一边渴望"],
    "safe": ["安心", "心里暖暖的", "被珍惜着"],
    "guilty_pleasure": ["不可以这样……可是……", "明明是兄妹，可是……", "这样是不对的"],
    "surprised": ["吓了一跳", "没料到", "没想到"],
    "embarrassed": ["太丢人了", "羞死人了", "好想找个地缝钻进去"],
    "floating": ["飘飘然", "像在云端", "整个人轻飘飘的"],
    "surrender": ["不想抵抗了", "随你怎么样吧", "交给你了"],
    "loved": ["被爱着", "心里好满", "好喜欢你"],
}

ACTION_URGES = {
    "pull_closer": ["想贴近", "不自觉地贴上去", "往你怀里蹭", "伸手环住你的脖子"],
    "push_away": ["想推开", "下意识地躲", "别过脸去"],
    "grab": ["手抓紧你的衣服", "手指攥紧床单", "抓着你的背"],
    "arch_into": ["腰不自觉地迎上去", "往你手心里蹭", "主动贴上来"],
    "hide_face": ["把脸埋起来", "埋进你颈窝", "别过脸不让你看"],
    "brace": ["手撑着", "扶住你的肩膀", "抓紧床头"],
    "wrap_legs": ["腿不自觉地缠上来", "腿勾住你的腰"],
    "cover_mouth": ["手捂住嘴", "咬着嘴唇不敢出声"],
}

VOCALIZATIONS = {
    "gasp": ["啊……", "唔……", "嘶……", "嗯……"],
    "moan_soft": ["小声地嘤咛", "闷哼一声", "细细的喘息声"],
    "moan_loud": ["忍不住叫出声", "发出一声甜腻的呻吟"],
    "whimper": ["呜咽一声", "小声地哼唧", "带着哭腔的哼声"],
    "broken": ["断断续续的喘息", "声音抖得不成样子", "破碎的呻吟"],
    "suppressed": ["咬着唇忍住", "捂着嘴不让自己出声", "死死憋着声音"],
    "breath_shake": ["呼吸在抖", "喘息声颤巍巍的"],
    "name_call": ["下意识叫出你的名字", "带着哭腔喊你的名字"],
}

COGNITIVE_SNIPPETS = {
    "blank": ["大脑一片空白", "什么都想不了了", "没法思考了"],
    "taboo_thought": ["是兄妹啊……", "这样是不对的……", "可是好舒服……"],
    "self_reproach": ["我在干什么……", "我怎么可以这样……", "好丢人……"],
    "awareness": ["意识到自己在发出声音", "意识到自己有多湿", "身体不受控制了"],
    "anticipation": ["接下来会怎么样……", "要进来了吗……", "会被发现吗……"],
    "sensation_focus": ["全部注意力都集中在那一点上", "只感觉得到你的触碰"],
    "intrusive_love": ["好喜欢你……", "不想让你停下来", "想要一直这样"],
}

BODY_AWARENESS = {
    "wet": ["感觉到自己湿透了", "那里已经湿得一塌糊涂", "内裤都湿了"],
    "aroused": ["身体热得不像话", "那里已经硬起来了", "乳头硬得发疼"],
    "flushed": ["知道自己脸红了", "耳朵肯定红透了"],
    "trembling": ["发现自己在抖", "手在颤抖", "腿软得撑不住"],
    "sensitive": ["那里变得好敏感", "碰一下都要跳起来"],
    "no_control": ["身体不听使唤了", "不受控制地扭着腰", "止不住地痉挛"],
}


# ============ 生成器 ============

@dataclass
class SensationReport:
    """一次beat/动作后的感受报告"""
    touch: List[str] = field(default_factory=list)
    temperature: List[str] = field(default_factory=list)
    physical: List[str] = field(default_factory=list)
    pleasure: List[str] = field(default_factory=list)
    emotion: List[str] = field(default_factory=list)
    action_urge: List[str] = field(default_factory=list)
    vocal: List[str] = field(default_factory=list)
    cognitive: List[str] = field(default_factory=list)
    body_aware: List[str] = field(default_factory=list)
    dominant_sensation: str = ""  # 最强烈的那一个感受
    summary: str = ""  # 一句话状态摘要

    def all_phrases(self) -> List[str]:
        return (self.touch + self.temperature + self.physical + self.pleasure +
                self.emotion + self.action_urge + self.vocal +
                self.cognitive + self.body_aware)

    def to_prompt_text(self) -> str:
        """转换为给模型的自然语言prompt"""
        lines = []
        if self.physical:
            lines.append("【身体反应】" + "，".join(self.physical[:3]) + "。")
        if self.touch:
            lines.append("【触感】" + "，".join(self.touch[:2]) + "。")
        if self.pleasure:
            lines.append("【快感】" + "，".join(self.pleasure[:2]) + "。")
        if self.emotion:
            lines.append("【情绪】" + "，".join(self.emotion[:2]) + "。")
        if self.action_urge:
            lines.append("【冲动】" + "，".join(self.action_urge[:1]) + "。")
        if self.vocal:
            lines.append("【声音】" + "，".join(self.vocal[:1]) + "。")
        if self.cognitive:
            lines.append("【脑中念头】" + "，".join(self.cognitive[:1]) + "。")
        return "\n".join(lines)


class SensationGenerator:
    """感受生成器"""
    def __init__(self):
        self.recently_used: Dict[str, List[str]] = {}  # 维度 -> 最近用过的短语
        self.history_size: int = 4  # 记忆最近N条避免重复

    def _sample(self, options: List[str], dimension: str,
                count: int = 1, avoid_recent: bool = True) -> List[str]:
        """从选项中采样，避免最近用过的"""
        if not options:
            return []
        recent = self.recently_used.get(dimension, [])
        available = options
        if avoid_recent and len(options) > len(recent) + 1:
            available = [o for o in options if o not in recent]
            if not available:
                available = options
        chosen = []
        pool = available[:]
        for _ in range(min(count, len(pool))):
            pick = random.choice(pool)
            chosen.append(pick)
            pool.remove(pick)
        # 更新recent
        recent.extend(chosen)
        self.recently_used[dimension] = recent[-self.history_size:]
        return chosen

    def _sample_dict(self, d: Dict[str, List[str]], dimension: str,
                     weight_key: Optional[str] = None, count: int = 1) -> List[str]:
        if weight_key and weight_key in d:
            return self._sample(d[weight_key], dimension, count)
        # 随机选一个子类别再采样
        key = random.choice(list(d.keys()))
        return self._sample(d[key], dimension, count)

    def generate(self, state, world, action_type: str = "",
                target_parts: List[str] = None, beat=None) -> SensationReport:
        """根据当前状态生成感受报告"""
        report = SensationReport()
        arousal = state.global_arousal
        ans = state.ans
        emotion = state.emotion
        mind = state.mind
        orgasm_phase = state.orgasm.phase
        pleasure_level = emotion.blend.get("pleasure", 0)
        shame = emotion.blend.get("shame", 0) + emotion.blend.get("embarrassment", 0)
        fear = emotion.blend.get("fear", 0)
        conflict = emotion.conflict_level
        trust = state.relationship_trust

        # === 1. 触感采样 ===
        if action_type:
            touch_opts = []
            # 根据动作类型
            if action_type in ["kiss", "deep_kiss"]:
                touch_opts += TOUCH_SENSATIONS["soft"]
                if arousal > 0.5:
                    touch_opts += TOUCH_SENSATIONS["electric"]
            if action_type in ["rub_clit", "finger_clit", "lick_clit", "suck_clit",
                                "rub_clit_through_panties"]:
                touch_opts += TOUCH_SENSATIONS["clitoris"]
                touch_opts += TOUCH_SENSATIONS["rhythmic"]
                if arousal > 0.6:
                    touch_opts += TOUCH_SENSATIONS["throbbing"]
                if state.clothing.get_friction_mod("clitoris") > 0.2:
                    touch_opts += TOUCH_SENSATIONS["friction"]
            if action_type in ["fondle_breast", "suck_nipple", "touch_breast"]:
                touch_opts += TOUCH_SENSATIONS["nipple"]
                touch_opts += TOUCH_SENSATIONS["soft"]
            if action_type in ["thrust", "deep_thrust", "quick_thrust", "missionary",
                                "cowgirl", "doggy", "penetrate", "finger_insert_one",
                                "finger_insert_two"]:
                touch_opts += TOUCH_SENSATIONS["penetrating"]
                touch_opts += TOUCH_SENSATIONS["inside"]
                if ans.arousal_global > 0.5:
                    touch_opts += TOUCH_SENSATIONS["wet_slide"]
            if action_type in ["cunnilingus"]:
                touch_opts += TOUCH_SENSATIONS["soft"]
                touch_opts += TOUCH_SENSATIONS["wet_slide"]
                touch_opts += TOUCH_SENSATIONS["clitoris"]
            if action_type in ["neck_kiss", "ear_kiss", "suck_neck"]:
                touch_opts += TOUCH_SENSATIONS["soft"]
                touch_opts += TOUCH_SENSATIONS["tickling"]
                if arousal > 0.4:
                    touch_opts += TOUCH_SENSATIONS["electric"]
            if action_type in ["tease"] or (beat and beat.rhythm == "paused"):
                touch_opts += TOUCH_SENSATIONS["teasing"]
            if touch_opts:
                report.touch = self._sample(touch_opts, "touch", count=min(2, len(touch_opts)))

        # === 2. 温度 ===
        temp_opts = []
        if arousal > 0.4 or ans.heart_rate > 90:
            temp_opts += TEMPERATURE_SENSATIONS["flushing"]
        if arousal > 0.6:
            temp_opts += TEMPERATURE_SENSATIONS["hot"]
        if shame > 0.5:
            temp_opts += TEMPERATURE_SENSATIONS["flushing"]
        if temp_opts:
            report.temperature = self._sample(temp_opts, "temperature", 1)

        # === 3. 生理反应 ===
        phys_opts = []
        # 心跳
        if ans.heart_rate > 110:
            phys_opts += PHYSICAL_RESPONSES["heart"]["racing"]
        elif ans.heart_rate > 90:
            phys_opts += PHYSICAL_RESPONSES["heart"]["pounding"]
        elif ans.heart_rate > 80:
            phys_opts += PHYSICAL_RESPONSES["heart"]["fast"]
        # 呼吸
        if orgasm_phase == "orgasm":
            phys_opts += PHYSICAL_RESPONSES["breathing"]["gasping"]
        elif ans.breathing_rate > 25:
            phys_opts += PHYSICAL_RESPONSES["breathing"]["gasping"]
            phys_opts += PHYSICAL_RESPONSES["breathing"]["heavy"]
        elif ans.breathing_rate > 20:
            phys_opts += PHYSICAL_RESPONSES["breathing"]["fast"]
        elif emotion.blend.get("shock", 0) > 0.5 or (beat and beat.note == "deep_thrust"):
            phys_opts += PHYSICAL_RESPONSES["breathing"]["holding"]
        if emotion.suppressing_sounds and arousal > 0.5:
            phys_opts += PHYSICAL_RESPONSES["breathing"]["shallow"]
        # 肌肉
        if orgasm_phase == "orgasm" or state.orgasm.intensity > 0.7:
            phys_opts += PHYSICAL_RESPONSES["muscle"]["spasm"]
            phys_opts += PHYSICAL_RESPONSES["muscle"]["arch"]
        elif ans.tremor > 0.5:
            phys_opts += PHYSICAL_RESPONSES["muscle"]["tremble"]
        if arousal > 0.7:
            phys_opts += PHYSICAL_RESPONSES["muscle"]["weak"]
            phys_opts += PHYSICAL_RESPONSES["muscle"]["tense"]
        elif arousal > 0.4:
            phys_opts += PHYSICAL_RESPONSES["muscle"]["tremble"]
        # 皮肤
        if ans.skin_flush > 0.6:
            phys_opts += PHYSICAL_RESPONSES["skin"]["flush"]
        if arousal > 0.7:
            phys_opts += PHYSICAL_RESPONSES["skin"]["sweat"]
        if shame > 0.7 and arousal < 0.3:
            phys_opts += PHYSICAL_RESPONSES["skin"]["goosebump"]
        if phys_opts:
            report.physical = self._sample(phys_opts, "physical",
                                          count=min(3, len(phys_opts)))

        # === 4. 局部快感 ===
        pleas_opts = []
        if orgasm_phase == "orgasm":
            pleas_opts += LOCAL_PLEASURE["overwhelming"]
            pleas_opts += LOCAL_PLEASURE["waves"]
        elif orgasm_phase == "plateau":
            pleas_opts += LOCAL_PLEASURE["plateau"]
            if state.orgasm.plateau_timer > 15:
                pleas_opts += LOCAL_PLEASURE["overwhelming"]
            else:
                pleas_opts += LOCAL_PLEASURE["throbbing"]
        elif orgasm_phase == "resolution":
            pleas_opts += LOCAL_PLEASURE["spreading"]
        else:
            if pleasure_level > 0.7:
                pleas_opts += LOCAL_PLEASURE["spreading"]
                pleas_opts += LOCAL_PLEASURE["throbbing"]
            elif pleasure_level > 0.4:
                pleas_opts += LOCAL_PLEASURE["building"]
                pleas_opts += LOCAL_PLEASURE["throbbing"]
            elif arousal > 0.3:
                pleas_opts += LOCAL_PLEASURE["building"]
            if pleasure_level > 0.5 and arousal < 0.5:
                pleas_opts += LOCAL_PLEASURE["sharp"]
        if pleas_opts:
            report.pleasure = self._sample(pleas_opts, "pleasure",
                                          count=min(2, len(pleas_opts)))

        # === 5. 情绪感受 ===
        emo_opts = []
        if shame > 0.6 and trust < 0.8:
            emo_opts += EMOTIONAL_FEELINGS["embarrassed"]
        elif shame > 0.4:
            emo_opts += EMOTIONAL_FEELINGS["shy"]
        if shame > 0.3 and arousal > 0.5 and mind.taboo_thrill > 0.3:
            emo_opts += EMOTIONAL_FEELINGS["guilty_pleasure"]
        if conflict > 0.5:
            emo_opts += EMOTIONAL_FEELINGS["conflicted"]
        if pleasure_level > 0.6 and trust > 0.7:
            if mind.acceptance_level > 0.6:
                emo_opts += EMOTIONAL_FEELINGS["surrender"]
            if trust > 0.85:
                emo_opts += EMOTIONAL_FEELINGS["loved"]
        if arousal > 0.7 and mind.mind_blank > 0.5:
            emo_opts += EMOTIONAL_FEELINGS["floating"]
        if mind.surprised:
            emo_opts += EMOTIONAL_FEELINGS["surprised"]
        if trust > 0.6 and arousal < 0.3 and pleasure_level < 0.3:
            emo_opts += EMOTIONAL_FEELINGS["safe"]
        if arousal > 0.4 and pleasure_level > 0.3 and not emotion.suppressing_movements:
            emo_opts += EMOTIONAL_FEELINGS["want_more"]
        if emo_opts:
            report.emotion = self._sample(emo_opts, "emotion",
                                         count=min(2, len(emo_opts)))

        # === 6. 动作冲动 ===
        urge_opts = []
        if mind.active_resistance_will > 0.5 and fear > 0.3:
            urge_opts += ACTION_URGES["push_away"]
        else:
            if trust > 0.5 and arousal > 0.3 and shame < 0.7:
                urge_opts += ACTION_URGES["pull_closer"]
                if arousal > 0.6:
                    urge_opts += ACTION_URGES["arch_into"]
            if shame > 0.6:
                urge_opts += ACTION_URGES["hide_face"]
            if arousal > 0.5 and fear < 0.3:
                urge_opts += ACTION_URGES["grab"]
            if orgasm_phase == "plateau" or arousal > 0.7:
                urge_opts += ACTION_URGES["brace"]
                if action_type in ["thrust", "missionary", "cowgirl", "doggy"]:
                    urge_opts += ACTION_URGES["wrap_legs"]
            if emotion.suppressing_sounds:
                urge_opts += ACTION_URGES["cover_mouth"]
        if urge_opts:
            report.action_urge = self._sample(urge_opts, "action_urge",
                                             count=min(2, len(urge_opts)))

        # === 7. 声音 ===
        voc_opts = []
        if emotion.suppressing_sounds:
            voc_opts += VOCALIZATIONS["suppressed"]
            if arousal > 0.5:
                voc_opts += VOCALIZATIONS["breath_shake"]
        else:
            if orgasm_phase == "orgasm":
                voc_opts += VOCALIZATIONS["broken"]
                voc_opts += VOCALIZATIONS["moan_loud"]
            elif arousal > 0.7:
                voc_opts += VOCALIZATIONS["moan_soft"]
                voc_opts += VOCALIZATIONS["whimper"]
                if mind.mind_blank > 0.6:
                    voc_opts += VOCALIZATIONS["name_call"]
            elif arousal > 0.5:
                if pleasure_level > 0.5:
                    voc_opts += VOCALIZATIONS["moan_soft"]
                voc_opts += VOCALIZATIONS["breath_shake"]
            elif arousal > 0.3 or ans.heart_rate > 90:
                voc_opts += VOCALIZATIONS["gasp"]
            if beat and beat.note == "deep_thrust" and arousal > 0.5:
                voc_opts += VOCALIZATIONS["gasp"]
        if world and world.privacy_level < 0.4 and arousal > 0.4:
            voc_opts = [v for v in voc_opts if v not in VOCALIZATIONS["moan_loud"]]
            voc_opts += VOCALIZATIONS["suppressed"]
        if voc_opts:
            report.vocal = self._sample(voc_opts, "vocal", 1)

        # === 8. 认知碎片 ===
        cog_opts = []
        if mind.mind_blank > 0.7 or orgasm_phase == "orgasm":
            cog_opts += COGNITIVE_SNIPPETS["blank"]
        if mind.taboo_thrill > 0.5 and shame > 0.4:
            cog_opts += COGNITIVE_SNIPPETS["taboo_thought"]
            if shame > 0.7:
                cog_opts += COGNITIVE_SNIPPETS["self_reproach"]
        if mind.self_consciousness > 0.6 and arousal > 0.4:
            cog_opts += COGNITIVE_SNIPPETS["awareness"]
        if arousal > 0.6:
            cog_opts += COGNITIVE_SNIPPETS["sensation_focus"]
        if trust > 0.8 and pleasure_level > 0.7 and shame < 0.5:
            cog_opts += COGNITIVE_SNIPPETS["intrusive_love"]
        if world and world.events.is_interrupted():
            cog_opts += COGNITIVE_SNIPPETS["anticipation"]
        if cog_opts:
            report.cognitive = self._sample(cog_opts, "cognitive", 1)

        # === 9. 身体自我察觉 ===
        aware_opts = []
        if arousal > 0.7 and state.get_wetness() > 0.4:
            aware_opts += BODY_AWARENESS["wet"]
        if arousal > 0.5 and shame > 0.5:
            aware_opts += BODY_AWARENESS["flushed"]
        if ans.tremor > 0.5:
            aware_opts += BODY_AWARENESS["trembling"]
        if orgasm_phase == "plateau":
            for pname in ["clitoris", "nipple_left", "nipple_right"]:
                part = None
                for region in state.body.values():
                    from .body.parts import BodyRegion
                    if isinstance(region, BodyRegion) and pname in region.sub_parts:
                        part = region.get(pname)
                        break
                if part and part.arousal > 0.8:
                    aware_opts += BODY_AWARENESS["sensitive"]
                    break
        if orgasm_phase == "orgasm" or state.orgasm.intensity > 0.8:
            aware_opts += BODY_AWARENESS["no_control"]
        if aware_opts:
            report.body_aware = self._sample(aware_opts, "body_aware", 1)

        # === 主导感受 ===
        all_sources = [
            (report.physical, "physical"), (report.pleasure, "pleasure"),
            (report.emotion, "emotion"), (report.touch, "touch"),
        ]
        for plist, ptype in all_sources:
            if plist:
                report.dominant_sensation = plist[0]
                break
        # 一句话摘要
        report.summary = state.get_state_summary()
        return report
