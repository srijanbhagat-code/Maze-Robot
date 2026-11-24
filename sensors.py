"""
sensors.py — Raw hardware abstractions for the maze robot.

Three distance sensors (FRONT, LEFT, RIGHT), two wheel encoders,
and a 3-axis accelerometer.  Everything is deliberately unitless at
this layer; the caller decides what "ticks" and "counts" mean.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from collections import deque
from typing import Deque


# ─────────────────────────────────────────────
#  Low-pass filter (exponential moving average)
# ─────────────────────────────────────────────

class EMA:
    """Exponential Moving Average — smooths noisy sensor readings."""

    def __init__(self, alpha: float = 0.2, seed: float = 0.0) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.value = seed

    def update(self, raw: float) -> float:
        self.value = self.alpha * raw + (1.0 - self.alpha) * self.value
        return self.value

    def reset(self, seed: float = 0.0) -> None:
        self.value = seed


# ─────────────────────────────────────────────
#  Sensor packets (immutable snapshots)
# ─────────────────────────────────────────────

@dataclass(frozen=True)
class DistanceReading:
    """
    Distances in metres (or whatever physical unit the hardware uses).
    A reading of SENSOR_MAX means "no obstacle detected in range".
    """
    front: float
    left: float
    right: float
    SENSOR_MAX: float = 2.0          # clip at 2 m; beyond = open space

    def clamp(self) -> "DistanceReading":
        return DistanceReading(
            min(self.front, self.SENSOR_MAX),
            min(self.left,  self.SENSOR_MAX),
            min(self.right, self.SENSOR_MAX),
            self.SENSOR_MAX,
        )


@dataclass(frozen=True)
class EncoderReading:
    """
    Raw cumulative encoder tick counts (unsigned, wrapping handled upstream).
    left_delta / right_delta = ticks since the previous reading.
    """
    left_delta: int
    right_delta: int


@dataclass(frozen=True)
class AccelReading:
    """
    Linear acceleration in m/s² along the robot's local axes.
      x  = forward/backward
      y  = left/right
      z  = vertical (gravity ~9.81 m/s²; used as a tilt check)
    """
    x: float
    y: float
    z: float

    @property
    def lateral(self) -> float:
        """Side-slip acceleration — non-zero usually means wheel-spin or a bump."""
        return abs(self.y)

    @property
    def tilt_deg(self) -> float:
        """Approximate forward tilt in degrees (0° = perfectly level)."""
        g = math.hypot(self.x, self.z)
        if g < 1e-6:
            return 0.0
        return math.degrees(math.asin(max(-1.0, min(1.0, self.x / g))))


@dataclass
class SensorFrame:
    """
    A single tick's worth of sensor data.
    All three sources are required; missing any means the tick is dropped.
    """
    distance: DistanceReading
    encoder: EncoderReading
    accel: AccelReading
    timestamp: float = 0.0           # seconds since boot; 0 = unknown


# ─────────────────────────────────────────────
#  Smoothed sensor hub
# ─────────────────────────────────────────────

class SensorHub:
    """
    Wraps raw sensor frames with light filtering.

    * Distance readings pass through per-channel EMA filters.
    * Encoder deltas are validated (sign-flip guard for wrap-around).
    * Accel lateral spike detection flags possible stuck/collision events.

    Usage
    -----
        hub = SensorHub()
        frame = hub.ingest(raw_distance, raw_encoder, raw_accel, t)
        # frame.distance values are already smoothed
    """

    MAX_ENCODER_JUMP = 500   # ticks: anything larger is treated as wrap-around

    def __init__(self, ema_alpha: float = 0.25) -> None:
        self._f  = EMA(ema_alpha)
        self._l  = EMA(ema_alpha)
        self._r  = EMA(ema_alpha)
        self._prev_enc_left  = 0
        self._prev_enc_right = 0
        self._enc_initialized = False
        self._lateral_history: Deque[float] = deque(maxlen=5)

    # ------------------------------------------------------------------
    def ingest(
        self,
        dist: DistanceReading,
        enc_abs_left: int,
        enc_abs_right: int,
        accel: AccelReading,
        timestamp: float = 0.0,
    ) -> SensorFrame:
        """
        Convert absolute encoder counts to deltas, smooth distances,
        and return a validated SensorFrame.
        """
        # Encoder delta with wrap-around guard
        if not self._enc_initialized:
            dl, dr = 0, 0
            self._enc_initialized = True
        else:
            dl = enc_abs_left  - self._prev_enc_left
            dr = enc_abs_right - self._prev_enc_right
            # Guard: if the jump is huge, assume encoder wrap or glitch
            if abs(dl) > self.MAX_ENCODER_JUMP:
                dl = 0
            if abs(dr) > self.MAX_ENCODER_JUMP:
                dr = 0
        self._prev_enc_left  = enc_abs_left
        self._prev_enc_right = enc_abs_right

        # Distance smoothing
        clamped = dist.clamp()
        smooth = DistanceReading(
            self._f.update(clamped.front),
            self._l.update(clamped.left),
            self._r.update(clamped.right),
        )

        # Lateral spike tracking
        self._lateral_history.append(accel.lateral)

        return SensorFrame(
            distance=smooth,
            encoder=EncoderReading(dl, dr),
            accel=accel,
            timestamp=timestamp,
        )

    @property
    def lateral_spike(self) -> bool:
        """True if recent lateral accel is unusually high (bump / slip)."""
        if len(self._lateral_history) < 3:
            return False
        avg = sum(self._lateral_history) / len(self._lateral_history)
        return avg > 1.5   # m/s² threshold -- tune to your hardware
