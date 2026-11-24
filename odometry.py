"""
odometry.py — Differential-drive dead-reckoning pose estimator.

State:  (x, y, theta)
         x, y   — metres from start
         theta  — heading in radians, 0 = initial forward direction
                  increases counter-clockwise (standard math convention)

The estimator runs two complementary estimates:
  1. Wheel odometry   — high-frequency, drifts slowly
  2. Accel integration — catches sudden angular events (bumps, slips)

They are blended with a simple alpha-trim weight: odometry dominates
during normal travel, the accelerometer corrects large heading
discrepancies when lateral g is detected.

Maths (differential drive kinematics)
--------------------------------------
Given left/right wheel travel distances d_L and d_R:

    d_centre = (d_R + d_L) / 2
    d_theta  = (d_R - d_L) / WHEEL_BASE

    if |d_theta| < eps (straight line):
        dx = d_centre * cos(theta)
        dy = d_centre * sin(theta)
    else (arc):
        R    = d_centre / d_theta          # turning radius
        dx   = R * (sin(theta + d_theta) - sin(theta))
        dy   = -R * (cos(theta + d_theta) - cos(theta))

    theta += d_theta
    x     += dx
    y     += dy
"""

from __future__ import annotations
import math
from dataclasses import dataclass

from sensors import SensorFrame


# ─────────────────────────────────────────────
#  Robot pose
# ─────────────────────────────────────────────

@dataclass
class Pose:
    x: float = 0.0          # metres east of start
    y: float = 0.0          # metres north of start
    theta: float = 0.0      # radians, CCW from east

    def __repr__(self) -> str:
        return (
            f"Pose(x={self.x:+.3f} m, y={self.y:+.3f} m, "
            f"θ={math.degrees(self.theta):+.1f}°)"
        )

    def copy(self) -> "Pose":
        return Pose(self.x, self.y, self.theta)


# ─────────────────────────────────────────────
#  Pose estimator
# ─────────────────────────────────────────────

class OdometryEstimator:
    """
    Tracks robot pose from encoder ticks and accel cross-checks.

    Parameters
    ----------
    wheel_radius_m  : metres per encoder tick (circumference / ticks_per_rev)
    wheel_base_m    : distance between left and right contact points
    accel_alpha     : blending weight for accel-based heading correction
                      (0 = ignore accel; 1 = trust only accel)
    dt              : fixed tick period in seconds (used for accel integration)
    """

    def __init__(
        self,
        wheel_radius_m: float = 0.034,   # 34 mm radius, typical small bot
        wheel_base_m:   float = 0.150,   # 150 mm between wheels
        accel_alpha:    float = 0.05,    # gentle correction weight
        dt:             float = 0.05,    # 20 Hz tick rate
    ) -> None:
        self._r    = wheel_radius_m
        self._b    = wheel_base_m
        self._aa   = accel_alpha
        self._dt   = dt

        # Metres per encoder tick
        self._m_per_tick = 2.0 * math.pi * self._r / 360.0   # 360 ticks/rev default

        self.pose  = Pose()
        self._omega_accel = 0.0          # running angular velocity from accel (rad/s)
        self._confidence  = 1.0          # 1 = high; drops near slips/collisions

    # ------------------------------------------------------------------
    def set_ticks_per_revolution(self, tpr: float) -> None:
        """Call this if you know the actual encoder resolution."""
        self._m_per_tick = 2.0 * math.pi * self._r / tpr

    # ------------------------------------------------------------------
    def update(self, frame: SensorFrame) -> Pose:
        """
        Consume one SensorFrame, update internal pose, return updated Pose.
        """
        enc = frame.encoder
        acc = frame.accel

        # ── Wheel distances (metres) ───────────────────────────────────
        d_L = enc.left_delta  * self._m_per_tick
        d_R = enc.right_delta * self._m_per_tick

        d_c     = (d_L + d_R) * 0.5            # arc length of centre point
        d_theta = (d_R - d_L) / self._b        # heading change from wheels

        # ── Accel-based angular velocity (yaw rate from lateral accel) ─
        # For a wheeled robot on a flat surface, lateral accel ≈ v * omega
        # omega_est = accel.y / v  (approximation; degrade to 0 when slow)
        speed  = abs(d_c) / self._dt if self._dt > 0 else 0.0
        if speed > 0.01:                       # only meaningful when moving
            omega_accel_raw = -acc.y / speed   # sign: left turn = +omega
        else:
            omega_accel_raw = 0.0

        # Blend accel correction gently into heading
        self._omega_accel = (
            (1.0 - self._aa) * self._omega_accel
            + self._aa * omega_accel_raw
        )
        accel_dtheta = self._omega_accel * self._dt

        # Weighted fusion: trust wheels more, use accel as sanity check
        fused_dtheta = (1.0 - self._aa) * d_theta + self._aa * accel_dtheta

        # ── Dead-reckoning position update ─────────────────────────────
        theta_old = self.pose.theta

        if abs(fused_dtheta) < 1e-9:           # essentially straight
            dx = d_c * math.cos(theta_old)
            dy = d_c * math.sin(theta_old)
        else:
            R  = d_c / fused_dtheta            # instantaneous turn radius
            dx = R * (math.sin(theta_old + fused_dtheta) - math.sin(theta_old))
            dy = -R * (math.cos(theta_old + fused_dtheta) - math.cos(theta_old))

        self.pose.x     += dx
        self.pose.y     += dy
        self.pose.theta  = _wrap(theta_old + fused_dtheta)

        # ── Confidence decay on large lateral spikes ───────────────────
        if acc.lateral > 2.0:                  # m/s² — likely a bump/slip
            self._confidence = max(0.2, self._confidence - 0.1)
        else:
            self._confidence = min(1.0, self._confidence + 0.01)

        return self.pose.copy()

    @property
    def confidence(self) -> float:
        """
        0.0–1.0 estimate of how trustworthy the current pose is.
        Drops after collisions or wheel slip; recovers slowly.
        """
        return self._confidence


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _wrap(angle: float) -> float:
    """Wrap angle into [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi
