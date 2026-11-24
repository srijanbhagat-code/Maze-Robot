"""
brain.py — Behavior engine for the maze robot.

Architecture
============

The brain is a Finite State Machine layered on top of a modified
right-hand-wall-follow rule, augmented with frontier-based exploration
to ensure the whole maze is discovered rather than just one route through.

States
------
  FORWARD          Normal travel; hug the right wall at a target gap.
  ALIGN_RIGHT      Adjust heading to maintain right-wall clearance.
  TURN_LEFT        Obstacle ahead; initiate left turn.
  TURN_RIGHT       Right wall lost; curve right to reacquire it.
  BACK_UP          Collision or imminent impact; reverse briefly.
  STUCK_SPIN       Encoder stall detected; spin to break free.
  EXPLORE_TARGET   Actively heading toward a specific frontier cell.
  DONE             Exploration complete (no more frontiers).

Motor Commands
--------------
The brain emits MotorCommand(left_pwm, right_pwm) each tick.
PWM is in [-1, 1] where:
  +1 = full forward
  -1 = full reverse
   0 = coast

The actual conversion to hardware PWM counts is your driver's job.
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

from sensors import SensorFrame
from odometry import Pose, OdometryEstimator
from mapping import OccupancyGrid, Cell


# ─────────────────────────────────────────────
#  Motor command
# ─────────────────────────────────────────────

@dataclass
class MotorCommand:
    """Normalised wheel velocities in [-1.0, +1.0]."""
    left:  float
    right: float

    def clamp(self) -> "MotorCommand":
        return MotorCommand(
            max(-1.0, min(1.0, self.left)),
            max(-1.0, min(1.0, self.right)),
        )

    @classmethod
    def stop(cls) -> "MotorCommand":
        return cls(0.0, 0.0)

    @classmethod
    def forward(cls, speed: float = 0.5) -> "MotorCommand":
        return cls(speed, speed)

    @classmethod
    def spin_left(cls, speed: float = 0.4) -> "MotorCommand":
        return cls(-speed, speed)

    @classmethod
    def spin_right(cls, speed: float = 0.4) -> "MotorCommand":
        return cls(speed, -speed)

    @classmethod
    def curve(cls, base: float, turn: float) -> "MotorCommand":
        """Curve: base = forward speed, turn = steering (-1=hard left, +1=hard right)."""
        return cls(base - turn * base, base + turn * base).clamp()


# ─────────────────────────────────────────────
#  Brain states
# ─────────────────────────────────────────────

class State(Enum):
    FORWARD         = auto()
    ALIGN_RIGHT     = auto()
    TURN_LEFT       = auto()
    TURN_RIGHT      = auto()
    BACK_UP         = auto()
    STUCK_SPIN      = auto()
    EXPLORE_TARGET  = auto()
    DONE            = auto()


# ─────────────────────────────────────────────
#  Brain
# ─────────────────────────────────────────────

class Brain:
    """
    The main decision loop.  Call tick() with each new SensorFrame;
    it returns a MotorCommand.

    Parameters
    ----------
    odom    : OdometryEstimator instance (shared with caller if desired)
    grid    : OccupancyGrid instance (same)
    right_wall_target_m : preferred clearance from right wall in metres
    forward_stop_m      : stop/turn when front sensor is below this
    base_speed          : normal forward PWM (0–1)
    turn_speed          : in-place spin PWM
    """

    # ── Tunables ──────────────────────────────────────────────────────
    STALL_TICKS           = 30     # ticks of zero encoder = stall
    BACK_UP_TICKS         = 15     # ticks to reverse after collision
    STUCK_SPIN_TICKS      = 20     # ticks of spin to break free
    TURN_TICKS            = 18     # ticks for a ~90° in-place turn (tune!)
    FRONTIER_ARRIVE_DIST  = 0.15   # metres — "close enough" to a target

    def __init__(
        self,
        odom: OdometryEstimator,
        grid: OccupancyGrid,
        right_wall_target_m: float = 0.18,
        forward_stop_m:      float = 0.15,
        base_speed:          float = 0.50,
        turn_speed:          float = 0.38,
    ) -> None:
        self._odom   = odom
        self._grid   = grid
        self._rw_tgt = right_wall_target_m
        self._fw_stp = forward_stop_m
        self._bspd   = base_speed
        self._tspd   = turn_speed

        self._state          = State.FORWARD
        self._state_ticks    = 0         # ticks spent in current state
        self._stall_counter  = 0         # ticks with no encoder movement
        self._target_cell: Optional[Cell] = None
        self._target_heading: Optional[float] = None   # for turning

        self._pose: Optional[Pose] = None

        # History for logging / debugging
        self.tick_count = 0
        self.state_log: list[tuple[int, State]] = []

    # ------------------------------------------------------------------
    #  Main entry point
    # ------------------------------------------------------------------

    def tick(self, frame: SensorFrame) -> MotorCommand:
        """Process one sensor frame; return motor command."""
        # ── Update pose and map ────────────────────────────────────────
        self._pose = self._odom.update(frame)
        d = frame.distance
        self._grid.update(self._pose, d.front, d.left, d.right)

        # ── Stall detection ───────────────────────────────────────────
        moving = (abs(frame.encoder.left_delta) + abs(frame.encoder.right_delta)) > 2
        if not moving:
            self._stall_counter += 1
        else:
            self._stall_counter = 0

        self._state_ticks += 1
        self.tick_count   += 1

        # ── State machine ─────────────────────────────────────────────
        cmd = self._run_state(frame)

        return cmd.clamp()

    # ------------------------------------------------------------------
    #  State machine dispatcher
    # ------------------------------------------------------------------

    def _run_state(self, frame: SensorFrame) -> MotorCommand:
        d    = frame.distance
        pose = self._pose

        # ── Hard interrupt: stall for too long → STUCK_SPIN ───────────
        if (
            self._state not in (State.STUCK_SPIN, State.BACK_UP, State.DONE)
            and self._stall_counter >= self.STALL_TICKS
        ):
            self._transition(State.STUCK_SPIN)

        # ── Hard interrupt: imminent front collision → BACK_UP ─────────
        if (
            self._state not in (State.BACK_UP, State.STUCK_SPIN, State.DONE)
            and d.front < 0.05
        ):
            self._transition(State.BACK_UP)

        # ── Dispatch ──────────────────────────────────────────────────
        s = self._state

        if s == State.DONE:
            return MotorCommand.stop()

        elif s == State.BACK_UP:
            return self._state_back_up(d)

        elif s == State.STUCK_SPIN:
            return self._state_stuck_spin(d)

        elif s == State.TURN_LEFT:
            return self._state_turn_left(d)

        elif s == State.TURN_RIGHT:
            return self._state_turn_right(d)

        elif s == State.EXPLORE_TARGET:
            return self._state_explore_target(d, pose)

        else:  # FORWARD / ALIGN_RIGHT — wall-following main loop
            return self._state_forward(d, pose)

    # ------------------------------------------------------------------
    #  Individual state handlers
    # ------------------------------------------------------------------

    def _state_forward(self, d, pose: Pose) -> MotorCommand:
        """Wall-following forward drive with right-wall PD control."""
        # ── Check for DONE (no more frontiers and we've explored well) ─
        if self._state_ticks % 50 == 0:          # check every 50 ticks
            if not self._grid.frontiers():
                self._transition(State.DONE)
                return MotorCommand.stop()

        # ── Front obstacle → turn left ─────────────────────────────────
        if d.front < self._fw_stp:
            self._transition(State.TURN_LEFT)
            return MotorCommand.spin_left(self._tspd)

        # ── Right-wall PD steering ────────────────────────────────────
        # Error: positive = too far from wall, negative = too close
        if d.right >= d.SENSOR_MAX * 0.95:      # right wall lost entirely
            # Curve gently right to find the wall again
            self._transition(State.TURN_RIGHT)
            return MotorCommand.curve(self._bspd * 0.6, 0.3)

        err = d.right - self._rw_tgt            # + = too far right
        # P-control: steer left when too close, right when too far
        steer = -0.8 * err                      # gain = 0.8
        steer = max(-0.5, min(0.5, steer))
        cmd   = MotorCommand.curve(self._bspd, steer)

        # Occasionally switch to frontier targeting if map growing
        if self._state_ticks % 200 == 0:
            target = self._grid.nearest_frontier(pose)
            if target is not None:
                self._target_cell = target
                self._transition(State.EXPLORE_TARGET)
                return cmd

        return cmd

    def _state_turn_left(self, d) -> MotorCommand:
        """Spin left until front is clear or max ticks elapsed."""
        if d.front > self._fw_stp * 1.8 or self._state_ticks >= self.TURN_TICKS * 2:
            self._transition(State.FORWARD)
        return MotorCommand.spin_left(self._tspd)

    def _state_turn_right(self, d) -> MotorCommand:
        """Curve right until right wall is detected or max ticks."""
        if d.right < self._rw_tgt * 1.5 or self._state_ticks >= self.TURN_TICKS:
            self._transition(State.FORWARD)
        return MotorCommand.curve(self._bspd * 0.5, 0.5)

    def _state_back_up(self, d) -> MotorCommand:
        """Reverse for a fixed number of ticks."""
        if self._state_ticks >= self.BACK_UP_TICKS:
            self._transition(State.TURN_LEFT)
        return MotorCommand(-self._bspd * 0.4, -self._bspd * 0.4)

    def _state_stuck_spin(self, d) -> MotorCommand:
        """Spin in place to escape a stuck position."""
        if self._state_ticks >= self.STUCK_SPIN_TICKS:
            self._stall_counter = 0
            self._transition(State.FORWARD)
        return MotorCommand.spin_left(self._tspd * 1.1)

    def _state_explore_target(self, d, pose: Pose) -> MotorCommand:
        """
        Head toward self._target_cell using simple bearing control.
        Falls back to FORWARD if target reached or obstacle encountered.
        """
        if self._target_cell is None:
            self._transition(State.FORWARD)
            return MotorCommand.forward(self._bspd)

        # Check arrival
        tx, ty   = self._grid.cell_to_world(self._target_cell)
        dist_rem = math.hypot(tx - pose.x, ty - pose.y)
        if dist_rem < self.FRONTIER_ARRIVE_DIST:
            self._target_cell = None
            self._transition(State.FORWARD)
            return MotorCommand.forward(self._bspd)

        # Check obstacle ahead
        if d.front < self._fw_stp:
            self._target_cell = None
            self._transition(State.TURN_LEFT)
            return MotorCommand.spin_left(self._tspd)

        # Compute desired heading toward target
        desired  = math.atan2(ty - pose.y, tx - pose.x)
        err      = _angle_diff(desired, pose.theta)   # in [-pi, pi]

        # P-controller on heading error
        steer    = -1.2 * err / math.pi               # normalised to [-1, 1]
        steer    = max(-0.7, min(0.7, steer))

        if abs(err) > math.radians(60):
            # Heading very wrong — spin in place first
            if err > 0:
                return MotorCommand.spin_left(self._tspd)
            else:
                return MotorCommand.spin_right(self._tspd)

        return MotorCommand.curve(self._bspd, steer).clamp()

    # ------------------------------------------------------------------
    #  State transition helper
    # ------------------------------------------------------------------

    def _transition(self, new_state: State) -> None:
        if new_state != self._state:
            self.state_log.append((self.tick_count, new_state))
            self._state       = new_state
            self._state_ticks = 0

    # ------------------------------------------------------------------
    #  Read-only properties for telemetry
    # ------------------------------------------------------------------

    @property
    def state(self) -> State:
        return self._state

    @property
    def pose(self) -> Optional[Pose]:
        return self._pose


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def _angle_diff(a: float, b: float) -> float:
    """Signed difference a - b, wrapped to [-pi, pi]."""
    d = a - b
    while d >  math.pi: d -= 2 * math.pi
    while d < -math.pi: d += 2 * math.pi
    return d
