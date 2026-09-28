"""
Layer 0b: 自主神经系统 (ANS)
控制全局生理反应：心跳、呼吸、潮红、肌张力等
"""
import math
from dataclasses import dataclass


@dataclass
class ANSState:
    sympathetic: float = 0.1       # 交感神经兴奋度 0-1
    parasympathetic: float = 0.9   # 副交感
    arousal_global: float = 0.0    # 全局性兴奋度 0-1
    heart_rate: float = 75.0
    heart_rate_base: float = 75.0
    breathing_rate: float = 14.0
    breathing_depth: str = "normal"  # normal/shallow/deep/gasping/holding
    breathing_rhythm: str = "regular"
    skin_flush: float = 0.0
    skin_temperature: float = 36.5
    skin_sweat: float = 0.0
    saliva: float = 0.5
    muscle_tone: float = 0.2
    tremor: float = 0.0
    pupil_dilation: float = 0.0
    voice_breathiness: float = 0.0
    voice_tightness: float = 0.0
    dizzy: float = 0.0
    vision_blur: float = 0.0
    ears_ringing: float = 0.0

    def update_from_firing(self, fired_signals: dict, dt: float, orgasm_phase: str = None,
                           forbidden_bonus: float = 0.0):
        """从神经发放信号更新ANS"""
        # 性相关信号强度
        sexual_signal = 0.0
        sexual_parts = ["clitoris", "vaginal_canal", "g_spot", "nipple_left",
                       "nipple_right", "inner_labia", "inner_thighs"]
        for pname in sexual_parts:
            sexual_signal += fired_signals.get(pname, 0.0)
        sexual_signal = min(sexual_signal * 0.35, 1.5)
        # 禁果效应额外加成
        sexual_signal += forbidden_bonus * 0.5

        # 更新arousal_global（平滑上升，不要一步到顶）
        self.arousal_global += sexual_signal * 0.25
        self.arousal_global *= math.exp(-0.005 * dt)
        self.arousal_global = max(0.0, min(1.0, self.arousal_global))

        # 交感/副交感平衡
        target_sym = 0.1 + self.arousal_global * 0.8
        if orgasm_phase == "orgasm":
            target_sym = 1.0
        self.sympathetic += (target_sym - self.sympathetic) * 0.2
        self.parasympathetic = 1.0 - self.sympathetic * 0.9

        # 心率由 CharacterState.tick 在本轮所有生理、疲劳和情绪信号汇总后统一更新。
        # 呼吸
        target_br = 14 + self.arousal_global * 20
        if orgasm_phase == "orgasm":
            target_br = 42
            self.breathing_depth = "gasping"
            self.breathing_rhythm = "irregular"
        elif self.arousal_global > 0.7:
            self.breathing_depth = "shallow"
            self.breathing_rhythm = "panting"
        elif self.arousal_global > 0.4:
            self.breathing_depth = "shallow"
            self.breathing_rhythm = "regular"
        else:
            self.breathing_depth = "normal"
            self.breathing_rhythm = "regular"
        self.breathing_rate += (target_br - self.breathing_rate) * 0.2

        # 潮红
        target_flush = self.arousal_global * 0.8
        if orgasm_phase == "orgasm":
            target_flush = 1.0
        self.skin_flush += (target_flush - self.skin_flush) * 0.15

        # 体温
        self.skin_temperature += ((36.5 + self.arousal_global * 1.2) - self.skin_temperature) * 0.05 * dt

        # 出汗
        target_sweat = self.arousal_global * 0.5
        if orgasm_phase == "orgasm":
            target_sweat = 0.9
        self.skin_sweat += (target_sweat - self.skin_sweat) * 0.06 * dt

        # 肌张力（高潮时强直）
        target_tone = 0.2 + self.arousal_global * 0.5
        if orgasm_phase == "orgasm":
            target_tone = 1.0
        self.muscle_tone += (target_tone - self.muscle_tone) * 0.12 * dt

        # 颤抖
        target_tremor = self.arousal_global * 0.4
        if orgasm_phase == "orgasm":
            target_tremor = 1.0
        self.tremor += (target_tremor - self.tremor) * 0.1 * dt

        # 声音喘气程度
        self.voice_breathiness = self.arousal_global * 0.8
        self.voice_tightness = self.arousal_global * 0.5 + (0.5 if orgasm_phase == "orgasm" else 0)

        # 瞳孔
        self.pupil_dilation = self.sympathetic * 0.8

        # 高潮时眩晕/视线模糊/耳鸣
        if orgasm_phase == "orgasm":
            self.dizzy = min(1.0, self.dizzy + 0.3 * dt)
            self.vision_blur = min(1.0, self.vision_blur + 0.2 * dt)
            self.ears_ringing = min(0.8, self.ears_ringing + 0.15 * dt)
        else:
            self.dizzy *= math.exp(-0.1 * dt)
            self.vision_blur *= math.exp(-0.15 * dt)
            self.ears_ringing *= math.exp(-0.2 * dt)

    def reset(self):
        """重置到静息状态"""
        self.__init__()

    def to_dict(self) -> dict:
        return {
            "sympathetic": self.sympathetic,
            "parasympathetic": self.parasympathetic,
            "arousal_global": self.arousal_global,
            "heart_rate": self.heart_rate,
            "heart_rate_base": self.heart_rate_base,
            "breathing_rate": self.breathing_rate,
            "breathing_depth": self.breathing_depth,
            "breathing_rhythm": self.breathing_rhythm,
            "skin_flush": self.skin_flush,
            "skin_temperature": self.skin_temperature,
            "skin_sweat": self.skin_sweat,
            "saliva": self.saliva,
            "muscle_tone": self.muscle_tone,
            "tremor": self.tremor,
            "pupil_dilation": self.pupil_dilation,
            "voice_breathiness": self.voice_breathiness,
            "voice_tightness": self.voice_tightness,
            "dizzy": self.dizzy,
            "vision_blur": self.vision_blur,
            "ears_ringing": self.ears_ringing,
        }
