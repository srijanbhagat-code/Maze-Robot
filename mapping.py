"""
mapping.py — Sparse 2-D occupancy grid for real-time maze mapping.

Cells are indexed by (grid_col, grid_row) integer tuples.
Each cell holds a probability-like log-odds value:
  * Strongly negative  = free
  * 0                  = unknown (initial state)
  * Strongly positive  = occupied (wall)

The grid is lazily allocated (only cells observed so far exist in memory),
so even a large maze consumes little RAM until explored.

Frontier detection
------------------
A "frontier" is a free cell adjacent to an unknown cell.
The exploration strategy picks frontiers as next targets so the robot
systematically uncovers the maze.
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Set, Tuple

from odometry import Pose


# ─────────────────────────────────────────────
#  Types
# ─────────────────────────────────────────────

Cell = Tuple[int, int]


# ─────────────────────────────────────────────
#  Log-odds helpers
# ─────────────────────────────────────────────

_L_OCC  =  0.85    # log-odds hit   (sensor says wall)
_L_FREE = -0.40    # log-odds miss  (sensor passes through)
_L_MAX  =  5.00    # clamp ceiling
_L_MIN  = -5.00    # clamp floor
_L_THRESH_OCC  = 0.50   # report as occupied above this
_L_THRESH_FREE = -0.50  # report as free below this


def _l2prob(lo: float) -> float:
    """Log-odds to probability."""
    return 1.0 - 1.0 / (1.0 + math.exp(lo))


# ─────────────────────────────────────────────
#  Occupancy grid
# ─────────────────────────────────────────────

class OccupancyGrid:
    """
    Sparse occupancy grid.

    Parameters
    ----------
    resolution   : metres per grid cell (e.g. 0.05 = 5 cm cells)
    sensor_range : maximum reliable sensor range in metres
    """

    NEIGHBORS_4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    NEIGHBORS_8 = NEIGHBORS_4 + [(1,1),(-1,1),(1,-1),(-1,-1)]

    def __init__(
        self,
        resolution: float   = 0.05,
        sensor_range: float = 1.80,
    ) -> None:
        self.res          = resolution
        self.sensor_range = sensor_range
        self._grid: Dict[Cell, float] = {}  # cell → log-odds
        self._visited: Set[Cell] = set()    # cells the robot has been inside

    # ------------------------------------------------------------------
    #  Coordinate helpers
    # ------------------------------------------------------------------

    def world_to_cell(self, x: float, y: float) -> Cell:
        """Convert world coordinates (metres) to grid cell indices."""
        return (int(math.floor(x / self.res)), int(math.floor(y / self.res)))

    def cell_to_world(self, cell: Cell) -> Tuple[float, float]:
        """Return world coords of cell centre."""
        col, row = cell
        return ((col + 0.5) * self.res, (row + 0.5) * self.res)

    # ------------------------------------------------------------------
    #  Cell state queries
    # ------------------------------------------------------------------

    def log_odds(self, cell: Cell) -> float:
        return self._grid.get(cell, 0.0)

    def is_occupied(self, cell: Cell) -> bool:
        return self.log_odds(cell) >= _L_THRESH_OCC

    def is_free(self, cell: Cell) -> bool:
        return self.log_odds(cell) <= _L_THRESH_FREE

    def is_unknown(self, cell: Cell) -> bool:
        return not self.is_occupied(cell) and not self.is_free(cell)

    def probability(self, cell: Cell) -> float:
        return _l2prob(self.log_odds(cell))

    # ------------------------------------------------------------------
    #  Update from one distance sensor ray
    # ------------------------------------------------------------------

    def _update_cell(self, cell: Cell, lo_delta: float) -> None:
        current = self._grid.get(cell, 0.0)
        self._grid[cell] = max(_L_MIN, min(_L_MAX, current + lo_delta))

    def _raycast(
        self,
        ox: float, oy: float,    # sensor origin in world coords
        angle: float,             # ray direction (radians, world frame)
        hit_dist: float,          # measured distance (metres)
    ) -> None:
        """
        Mark cells along a ray as free, and the terminal cell as occupied.
        Uses Bresenham-like stepping in world space.
        """
        step = self.res * 0.5    # half-cell steps for accuracy
        dist = 0.0
        hit  = hit_dist < self.sensor_range  # did we actually detect a wall?

        while dist < min(hit_dist, self.sensor_range) - step:
            dist += step
            cx = ox + dist * math.cos(angle)
            cy = oy + dist * math.sin(angle)
            self._update_cell(self.world_to_cell(cx, cy), _L_FREE)

        if hit:
            ex = ox + hit_dist * math.cos(angle)
            ey = oy + hit_dist * math.sin(angle)
            self._update_cell(self.world_to_cell(ex, ey), _L_OCC)

    # ------------------------------------------------------------------
    #  Public update: call once per tick with pose + distance frame
    # ------------------------------------------------------------------

    def update(self, pose: Pose, front: float, left: float, right: float) -> None:
        """
        Integrate one set of distance readings at the given pose.

        Sensor geometry (local offsets from robot centre):
          - Front sensor: straight ahead
          - Left  sensor: 90° to the left
          - Right sensor: 90° to the right
        """
        sx, sy, th = pose.x, pose.y, pose.theta

        # Robot body cell is free and visited
        robot_cell = self.world_to_cell(sx, sy)
        self._update_cell(robot_cell, _L_FREE)
        self._visited.add(robot_cell)

        # Raycast each sensor (angle in world frame)
        self._raycast(sx, sy, th,             front)   # forward
        self._raycast(sx, sy, th + math.pi/2, left)    # left  (+90°)
        self._raycast(sx, sy, th - math.pi/2, right)   # right (-90°)

    # ------------------------------------------------------------------
    #  Frontier detection
    # ------------------------------------------------------------------

    def frontiers(self) -> List[Cell]:
        """
        Return all frontier cells (free cells adjacent to unknown cells),
        sorted by distance from grid origin (closest first).
        """
        result: List[Cell] = []
        for cell, lo in self._grid.items():
            if lo > _L_THRESH_FREE:          # skip occupied/unknown
                continue
            for dc, dr in self.NEIGHBORS_4:
                nb = (cell[0]+dc, cell[1]+dr)
                if self.is_unknown(nb):
                    result.append(cell)
                    break
        result.sort(key=lambda c: c[0]**2 + c[1]**2)
        return result

    # ------------------------------------------------------------------
    #  Nearest unvisited frontier (for navigation target selection)
    # ------------------------------------------------------------------

    def nearest_frontier(self, pose: Pose) -> Optional[Cell]:
        """Return the frontier cell nearest to the given pose, or None."""
        robot_cell = self.world_to_cell(pose.x, pose.y)
        fronts = self.frontiers()
        if not fronts:
            return None
        def dist2(c: Cell) -> float:
            return (c[0]-robot_cell[0])**2 + (c[1]-robot_cell[1])**2
        return min(fronts, key=dist2)

    # ------------------------------------------------------------------
    #  Simple ASCII debug dump (for terminal visualisation)
    # ------------------------------------------------------------------

    def ascii_map(self, pose: Optional[Pose] = None, radius: int = 20) -> str:
        """Render a small window around the robot as ASCII art."""
        if pose is None:
            cx, cy = 0, 0
        else:
            cx, cy = self.world_to_cell(pose.x, pose.y)

        lines = []
        for row in range(cy + radius, cy - radius - 1, -1):
            row_chars = []
            for col in range(cx - radius, cx + radius + 1):
                cell = (col, row)
                if pose is not None and cell == (cx, cy):
                    row_chars.append("R")
                elif self.is_occupied(cell):
                    row_chars.append("#")
                elif self.is_free(cell):
                    row_chars.append(".")
                else:
                    row_chars.append(" ")
            lines.append("".join(row_chars))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    #  Coverage stats
    # ------------------------------------------------------------------

    @property
    def total_cells(self) -> int:
        return len(self._grid)

    @property
    def visited_cells(self) -> int:
        return len(self._visited)

    @property
    def free_cells(self) -> int:
        return sum(1 for lo in self._grid.values() if lo <= _L_THRESH_FREE)

    @property
    def occupied_cells(self) -> int:
        return sum(1 for lo in self._grid.values() if lo >= _L_THRESH_OCC)
