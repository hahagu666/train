"""Deterministic physical load and stamina calculations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


# === 日志补接线（详细排障） ===
try:
    from app.logger import debug, info, success, warning, error, trace
except Exception:
    debug = info = success = warning = error = trace = lambda *a, **k: None
@dataclass
class StaminaState:
    capacity: float = 100.0
    current: float = 100.0
    fatigue: float = 0.0
    sleep_debt: float = 0.0
    pain: float = 0.0
    recovery_rate: float = 0.08
    exhaustion_threshold: float = 10.0

    def to_dict(self) -> dict:
        return {
            "capacity": self.capacity,
            "current": self.current,
            "fatigue": self.fatigue,
            "sleep_debt": self.sleep_debt,
            "pain": self.pain,
            "recovery_rate": self.recovery_rate,
            "exhaustion_threshold": self.exhaustion_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StaminaState":
        state = cls()
        for key, value in data.items():
            if hasattr(state, key):
                setattr(state, key, float(value))
        state.capacity = max(1.0, state.capacity)
        state.current = max(0.0, min(state.capacity, state.current))
        return state

    @property
    def ratio(self) -> float:
        return self.current / self.capacity if self.capacity else 0.0

    @property
    def exhausted(self) -> bool:
        return self.current <= self.exhaustion_threshold


def configure_stamina(body_params, appearance=None) -> StaminaState:
    base = float(getattr(body_params, "stamina", 1.0))
    height = float(getattr(appearance, "height_cm", None) or getattr(appearance, "height", 165))
    weight = float(getattr(appearance, "weight_kg", None) or max(45.0, height - 115.0))
    size_factor = max(0.75, min(1.35, (weight / 60.0) ** 0.18 * (height / 165.0) ** 0.08))
    capacity = max(40.0, min(160.0, 100.0 * base * size_factor))
    return StaminaState(capacity=capacity, current=capacity, recovery_rate=0.08 * max(0.6, min(1.5, base)))


def body_load_factor(character) -> float:
    """Estimate posture and movement load from objective body parameters."""
    appearance = getattr(character, "appearance", None)
    if appearance is None:
        return 1.0
    height = float(getattr(appearance, "height_cm", None) or getattr(appearance, "height", 165))
    weight = float(getattr(appearance, "weight_kg", None) or 60.0)
    body_type = str(getattr(appearance, "body_type", ""))
    factor = 1.0 + max(0.0, (weight - 60.0) / 180.0)
    factor += max(0.0, (height - 170.0) / 500.0)
    if any(term in body_type for term in ("纤细", "清瘦", "娇小")):
        factor += 0.05
    return max(0.8, min(1.35, factor))


def action_stamina_cost(action_type: str, duration: float, intensity: float = 0.7,
                        posture_load: float = 1.0, body_factor: float = 1.0,
                        fatigue: float = 0.0) -> float:
    base_rates = {
        "talk": 0.02, "wait": 0.01, "hug": 0.08, "kiss": 0.1,
        "touch_breast": 0.16, "fondle_breast": 0.2, "touch_over_clothes": 0.18,
        "rub_clit": 0.22, "finger_clit": 0.24, "cunnilingus": 0.3,
        "finger_insert_one": 0.24, "finger_insert_two": 0.28,
        "penetrate": 0.3, "thrust": 0.38, "slow_thrust": 0.28,
        "deep_thrust": 0.42, "quick_thrust": 0.48,
        "missionary": 0.32, "cowgirl": 0.4, "doggy": 0.42,
        "rest": -0.15, "aftercare": -0.08, "cuddle": -0.04,
    }
    rate = base_rates.get(action_type, 0.08)
    intensity_factor = max(0.5, min(1.8, 0.65 + float(intensity)))
    fatigue_factor = 1.0 + max(0.0, fatigue) * 0.8
    debug("身体引擎", f"体力消耗计算: action={action_type}, duration={duration:.1f}s, intensity={intensity:.2f}, 姿势={posture_load:.2f}, 体型={body_factor:.2f}, 疲劳={fatigue:.2f} -> rate={rate:.2f}")
    return max(-duration * 0.2, duration * rate * intensity_factor * posture_load * body_factor * fatigue_factor)


def apply_stamina_delta(stamina: StaminaState, delta: float) -> None:
    stamina.current = max(0.0, min(stamina.capacity, stamina.current - delta))
    if delta > 0:
        stamina.fatigue = min(1.0, stamina.fatigue + delta / max(stamina.capacity, 1.0) * 0.7)
        stamina.pain = min(1.0, stamina.pain + max(0.0, stamina.current <= stamina.exhaustion_threshold) * 0.02)
        debug("身体引擎", f"体力变化: delta={delta:.2f}, 当前={stamina.current:.1f}/{stamina.capacity:.1f}, 疲劳={stamina.fatigue:.2f}")


def recover_stamina(stamina: StaminaState, seconds: float, sleeping: bool = False) -> None:
    if seconds <= 0:
        return
    multiplier = 2.5 if sleeping else 1.0
    recovery = stamina.recovery_rate * seconds * multiplier * max(0.0, 1.0 - stamina.sleep_debt * 0.25)
    stamina.current = min(stamina.capacity, stamina.current + recovery)
    stamina.fatigue = max(0.0, stamina.fatigue - seconds / 3600.0 * (0.18 if sleeping else 0.05))
    stamina.pain = max(0.0, stamina.pain - seconds / 3600.0 * 0.12)
    if sleeping:
        stamina.sleep_debt = max(0.0, stamina.sleep_debt - seconds / 3600.0)
    else:
        stamina.sleep_debt = min(8.0, stamina.sleep_debt + seconds / 3600.0 * 0.02)
