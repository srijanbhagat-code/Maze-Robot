"""
robot.py — Top-level harness.

Two modes
---------
  1. SIMULATION (default)
     A software-in-the-loop (SIL) maze simulator.  The maze is generated
     randomly with recursive back-tracking.  Sensor readings are raycast
     from the robot's true position.  Run:

         python robot.py

  2. HARDWARE
     Replace the 'hardware_tick()' stub with your platform's sensor API,
     then call robot_loop(hardware=True).

Coordinate system
-----------------
  Origin = start position of the robot.
  +x  = initial forward direction.
  +y  = left of robot when facing +x.
  Angles in radians, CCW positive.
"""

from __future__ import annotations
import math
import random
import time
import sys
from typing import Optional

from sensors  import SensorHub, DistanceReading, AccelReading
from odometry import OdometryEstimator
from mapping  import OccupancyGrid
from brain    import Brain, MotorCommand


# ═══════════════════════════════════════════════════════════════════════
#  SIMULATION — Maze generator + physics model
# ═══════════════════════════════════════════════════════════════════════

class MazeSimulator:
    """
    A simple grid-maze simulator for software-in-the-loop testing.

    The maze is stored as a set of (col, row) wall cells on a fine grid.
    Robot physics:
      - Differential drive with noise.
      - Distance sensors as perfect raycasts + Gaussian noise.
      - Encoders as integrated wheel speeds + quantisation noise.
      - Accelerometer as derivative of velocity + noise.
    """

    CELL   = 0.40    # metres per maze cell (each cell is a passage or wall)
    WALL_T = 0.05    # wall thickness metres (half on each side of boundary)

    def __init__(
        self,
        cols: int = 15,
        rows: int = 15,
        seed: Optional[int] = None,
        wheel_base: float = 0.150,
        wheel_radius: float = 0.034,
        ticks_per_rev: int = 360,
    ) -> None:
        self._cols  = cols
        self._rows  = rows
        self._wb    = wheel_base
        self._wr    = wheel_radius
        self._tpr   = ticks_per_rev
        self._rng   = random.Random(seed)

        # Build maze walls via recursive back-tracking
        self._walls: set[tuple[int,int]] = set()
        self._generate_maze(cols, rows)

        # Robot true state
        self._x     = 0.5 * self.CELL        # start in cell (0,0) centre
        self._y     = 0.5 * self.CELL
        self._theta = 0.0                     # facing +x
        self._vl    = 0.0                     # left wheel linear velocity m/s
        self._vr    = 0.0                     # right wheel linear velocity m/s
        self._enc_l = 0                       # cumulative left encoder ticks
        self._enc_r = 0

        self._tick  = 0
        self._dt    = 0.05                    # simulation time step (s)

    # ------------------------------------------------------------------
    #  Maze generation (recursive back-tracking / DFS)
    # ------------------------------------------------------------------

    def _generate_maze(self, cols: int, rows: int) -> None:
        """Fill self._walls with occupied cell indices."""
        # Mark everything as wall first
        for c in range(cols):
            for r in range(rows):
                self._walls.add((c, r))

        # Carve passages via DFS
        visited: set[tuple[int,int]] = set()

        def carve(c: int, r: int) -> None:
            visited.add((c, r))
            self._walls.discard((c, r))          # this cell is passage
            dirs = [(2,0),(-2,0),(0,2),(0,-2)]
            self._rng.shuffle(dirs)
            for dc, dr in dirs:
                nc, nr = c+dc, r+dr
                if 0 <= nc < cols and 0 <= nr < rows and (nc,nr) not in visited:
                    # carve the wall between
                    self._walls.discard((c+dc//2, r+dr//2))
                    carve(nc, nr)

        # Start from top-left odd cell
        carve(0, 0)

        # Border walls
        for c in range(-1, cols+1):
            self._walls.add((c, -1))
            self._walls.add((c, rows))
        for r in range(-1, rows+1):
            self._walls.add((-1, r))
            self._walls.add((cols, r))

    # ------------------------------------------------------------------
    #  Raycast a single sensor beam
    # ------------------------------------------------------------------

    def _raycast(self, ox: float, oy: float, angle: float, max_dist: float = 2.0) -> float:
        """Return distance to nearest wall along the ray, up to max_dist."""
        step = 0.01   # 1 cm stepping
        d    = 0.0
        while d < max_dist:
            d   += step
            px   = ox + d * math.cos(angle)
            py   = oy + d * math.sin(angle)
            col  = int(math.floor(px / self.CELL))
            row  = int(math.floor(py / self.CELL))
            if (col, row) in self._walls:
                return d
        return max_dist

    # ------------------------------------------------------------------
    #  Step the physics and return sensor values
    # ------------------------------------------------------------------

    def step(self, cmd: MotorCommand) -> tuple[DistanceReading, int, int, AccelReading, float]:
        """
        Apply motor command, advance physics one tick.
        Returns (DistanceReading, enc_abs_left, enc_abs_right, AccelReading, timestamp).
        """
        MAX_SPEED = 0.5   # m/s at PWM=1.0

        # Target wheel velocities
        vl_tgt = cmd.left  * MAX_SPEED
        vr_tgt = cmd.right * MAX_SPEED

        # First-order lag (motor inertia tau = 3 ticks)
        tau = 0.15
        self._vl += (vl_tgt - self._vl) * (self._dt / tau)
        self._vr += (vr_tgt - self._vr) * (self._dt / tau)

        # Add small Gaussian noise
        nl = self._rng.gauss(0, 0.002)
        nr = self._rng.gauss(0, 0.002)

        # Differential drive kinematics
        d_L = (self._vl + nl) * self._dt
        d_R = (self._vr + nr) * self._dt

        d_c     = (d_L + d_R) * 0.5
        d_theta = (d_R - d_L) / self._wb

        theta_new = self._theta + d_theta
        if abs(d_theta) < 1e-9:
            dx = d_c * math.cos(self._theta)
            dy = d_c * math.sin(self._theta)
        else:
            R  = d_c / d_theta
            dx = R * (math.sin(theta_new) - math.sin(self._theta))
            dy = -R * (math.cos(theta_new) - math.cos(self._theta))

        # Collision check: don't enter walls
        nx, ny = self._x + dx, self._y + dy
        col_n = int(math.floor(nx / self.CELL))
        row_n = int(math.floor(ny / self.CELL))
        if (col_n, row_n) not in self._walls:
            self._x = nx
            self._y = ny
        else:
            # Hit a wall — zero velocities, small bounce back
            self._vl *= -0.1
            self._vr *= -0.1
        self._theta = (theta_new + math.pi) % (2 * math.pi) - math.pi

        # Encoder ticks (with quantisation)
        ticks_per_m = self._tpr / (2 * math.pi * self._wr)
        self._enc_l += int(d_L * ticks_per_m)
        self._enc_r += int(d_R * ticks_per_m)

        # Distance sensors (with small Gaussian noise)
        noise = lambda: self._rng.gauss(0, 0.01)
        dist = DistanceReading(
            front = max(0, self._raycast(self._x, self._y, self._theta)             + noise()),
            left  = max(0, self._raycast(self._x, self._y, self._theta + math.pi/2) + noise()),
            right = max(0, self._raycast(self._x, self._y, self._theta - math.pi/2) + noise()),
        )

        # Accelerometer (derivative of velocity + gravity, with noise)
        a_forward = (d_c / self._dt) * 0 + self._rng.gauss(0, 0.05)
        a_lateral = ((self._vr - self._vl) / self._wb) * (d_c / self._dt) + self._rng.gauss(0, 0.05)
        accel = AccelReading(x=a_forward, y=a_lateral, z=9.81 + self._rng.gauss(0, 0.02))

        self._tick += 1
        return dist, self._enc_l, self._enc_r, accel, self._tick * self._dt

    @property
    def true_x(self) -> float:   return self._x
    @property
    def true_y(self) -> float:   return self._y
    @property
    def true_theta(self) -> float: return self._theta


# ═══════════════════════════════════════════════════════════════════════
#  ROBOT LOOP
# ═══════════════════════════════════════════════════════════════════════

def build_robot(
    wheel_radius_m: float = 0.034,
    wheel_base_m:   float = 0.150,
    dt:             float = 0.05,
) -> tuple[SensorHub, OdometryEstimator, OccupancyGrid, Brain]:
    """Wire up all components and return them."""
    hub  = SensorHub(ema_alpha=0.25)
    odom = OdometryEstimator(
        wheel_radius_m = wheel_radius_m,
        wheel_base_m   = wheel_base_m,
        dt             = dt,
    )
    grid = OccupancyGrid(resolution=0.05, sensor_range=1.8)
    bot  = Brain(odom, grid)
    return hub, odom, grid, bot


def sim_loop(
    max_ticks:   int  = 5000,
    print_every: int  = 100,
    seed:        Optional[int] = 42,
    visualize:   bool = True,
) -> None:
    """Run the brain against the built-in SIL simulator."""

    print("=== MAZE ROBOT — Software-in-the-Loop Simulation ===\n")

    sim = MazeSimulator(cols=11, rows=11, seed=seed)
    hub, odom, grid, bot = build_robot()

    # Prime the sensor hub with the first reading (no delta yet)
    cmd = MotorCommand.forward(0.4)

    for tick in range(max_ticks):
        # 1. Step the simulated world forward
        dist_raw, enc_l, enc_r, accel_raw, ts = sim.step(cmd)

        # 2. Ingest through the sensor hub (smoothing + delta extraction)
        frame = hub.ingest(dist_raw, enc_l, enc_r, accel_raw, ts)

        # 3. Run the brain to get the next motor command
        cmd = bot.tick(frame)

        # 4. Print telemetry
        if tick % print_every == 0 or bot.state.name == "DONE":
            pose = bot.pose
            print(
                f"Tick {tick:5d} | State: {bot.state.name:<16} | "
                f"Pose: ({pose.x:+.2f}, {pose.y:+.2f}) θ={math.degrees(pose.theta):+5.1f}° | "
                f"True: ({sim.true_x:+.2f}, {sim.true_y:+.2f}) | "
                f"Conf: {odom.confidence:.2f} | "
                f"Cells: {grid.visited_cells} visited | "
                f"Dist F={frame.distance.front:.2f} L={frame.distance.left:.2f} R={frame.distance.right:.2f}"
            )
            if visualize and tick % (print_every * 5) == 0:
                print("\n" + grid.ascii_map(pose, radius=14) + "\n")

        if bot.state.name == "DONE":
            print(f"\n✓ Exploration complete at tick {tick}.")
            print(f"  Total cells mapped : {grid.total_cells}")
            print(f"  Free  cells        : {grid.free_cells}")
            print(f"  Occupied cells     : {grid.occupied_cells}")
            break
    else:
        print(f"\n✗ Max ticks ({max_ticks}) reached without full exploration.")
        print(f"  Frontiers remaining: {len(grid.frontiers())}")

    if visualize:
        print("\nFinal map:")
        print(grid.ascii_map(bot.pose, radius=20))


# ─────────────────────────────────────────────
#  Hardware stub — replace body with your API
# ─────────────────────────────────────────────

def hardware_tick_stub():
    """
    Replace this function with calls to your actual hardware:
      - Read your distance sensors → DistanceReading(front, left, right)
      - Read encoder counters      → enc_l, enc_r  (absolute cumulative)
      - Read accelerometer         → AccelReading(x, y, z)
      - Read clock                 → timestamp in seconds
    Return those five values.
    """
    raise NotImplementedError(
        "hardware_tick_stub() is not implemented. "
        "Wire up your sensor API here."
    )


def hardware_loop(dt: float = 0.05) -> None:
    """Drive a real robot. Replace hardware_tick_stub with your sensor API."""
    hub, odom, grid, bot = build_robot(dt=dt)
    cmd = MotorCommand.forward(0.4)

    while True:
        t_start = time.monotonic()

        dist_raw, enc_l, enc_r, accel_raw, ts = hardware_tick_stub()
        frame = hub.ingest(dist_raw, enc_l, enc_r, accel_raw, ts)
        cmd   = bot.tick(frame)

        # Send cmd.left, cmd.right to your motor driver here
        # e.g.  motor_driver.set_pwm(cmd.left, cmd.right)

        # Busy-wait to hold tick rate
        elapsed = time.monotonic() - t_start
        if elapsed < dt:
            time.sleep(dt - elapsed)


# ─────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    use_hardware = "--hardware" in sys.argv
    if use_hardware:
        hardware_loop()
    else:
        sim_loop(max_ticks=6000, print_every=50, seed=42, visualize=True)
