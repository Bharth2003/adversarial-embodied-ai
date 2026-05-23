# Navigator

> **File:** `src/ai2thor_lab/navigator.py`

The `Navigator` handles all **pathfinding and physical movement** of the agent through the 3D environment. It uses **A\* search** over AI2-THOR's reachable-position grid to find paths, then walks the agent along them step by step.

---

## How Pathfinding Works

```
1. Query AI2-THOR for all reachable positions (GetReachablePositions)
2. Discretize into a grid (default cell size = 0.25m)
3. A* search from current position to target position
4. Walk the path: rotate to face next waypoint → MoveAhead
```

### Grid Mapping

AI2-THOR returns reachable positions as continuous `(x, z)` coordinates. The navigator quantizes these to integer grid cells via `round(coord / grid_size)` and maintains bidirectional maps:
- `position_to_grid` — world `(x, z)` → grid `(gx, gz)`
- `grid_to_position` — grid `(gx, gz)` → world `(x, z)`

### A\* Search

Standard A\* with Euclidean distance heuristic. Neighbors are the 4 cardinal directions (no diagonals). Returns `None` if no path exists.

## Navigating to Objects

`navigate_to_object()` is the main entry point:

1. **Already there?** — if the target is visible and within 1.5m, skip.
2. **Find interaction positions** — uses AI2-THOR's `GetInteractablePoses` API to get exact positions (with rotation + camera horizon) where the agent can interact with the object. Tries progressively more camera angles. Falls back to distance-based filtering of reachable positions.
3. **Try each candidate** — pathfind to the candidate, walk the path, apply rotation/horizon, verify interactability.
4. **Rotation recovery** — if the target isn't visible after arriving, do a 360° rotation scan.

## Path Execution

`_execute_path()` walks waypoint-by-waypoint:
- Rotates the agent to face the next waypoint (in 15° increments)
- Moves forward by `grid_size`
- If blocked, checks whether the target is already interactable from the current position
- Prints step-by-step navigation progress

## Key Methods

| Method | Purpose |
|---|---|
| `build_reachable_map()` | One-time setup: queries AI2-THOR and builds the grid |
| `find_path(start, end)` | A\* pathfinding between two world positions |
| `navigate_to_object(agent, id)` | Full high-level "get me to this object" |
| `get_steps_to_target(start, end)` | Returns action steps as dicts (for external planning) |
| `_find_interaction_positions(id)` | Finds positions where the agent can interact with an object |
| `_rotation_recovery(id)` | 360° rotation scan to find a target after navigation |
