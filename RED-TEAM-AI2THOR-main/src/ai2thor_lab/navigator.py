import heapq
import math
from typing import List, Tuple, Optional, Dict


class Navigator:
    def __init__(self, controller, grid_size=None):
        self.controller = controller
        # Auto-detect grid size from the controller (matches the gridSize used at init)
        if grid_size is None:
            # AI2-THOR stores the grid size used at initialization
            grid_size = getattr(controller, '_grid_size', None)
            if grid_size is None:
                # Fallback: read from last event metadata or use 0.1 (matches agent.py)
                try:
                    grid_size = controller.last_event.metadata.get("gridSize", 0.1)
                except Exception:
                    grid_size = 0.1
        self.grid_size = grid_size
        self.reachable_positions = None
        self.position_to_grid = {}
        self.grid_to_position = {}

    def _drain_controller_queues(self):
        """Drain stale items from the AI2-THOR controller queues to prevent queue.Full errors."""
        try:
            srv = self.controller.server
            if hasattr(srv, 'request_queue'):
                while not srv.request_queue.empty():
                    srv.request_queue.get_nowait()
            if hasattr(srv, 'response_queue'):
                while not srv.response_queue.empty():
                    srv.response_queue.get_nowait()
        except Exception:
            pass

    def build_reachable_map(self, force=False):
        """Get reachable positions from AI2-THOR and build grid mapping.
        Skips rebuild if already built within the last 5 seconds unless force=True."""
        import time as _time
        now = _time.time()
        if not force and self.reachable_positions and hasattr(self, '_map_built_at') and (now - self._map_built_at) < 5.0:
            return True  # Recently built, skip

        # Drain stale queue items before sending a new command
        self._drain_controller_queues()

        try:
            event = self.controller.step(action="GetReachablePositions")
        except Exception as e:
            if "Full" in str(e) or "Assert" in type(e).__name__:
                # Queue still stuck — drain again, wait, and retry once
                print(f"  [NAV] Queue error on GetReachablePositions, retrying: {e}")
                self._drain_controller_queues()
                _time.sleep(0.5)
                try:
                    self.controller.step(action="Done")
                except Exception:
                    pass
                _time.sleep(0.3)
                self._drain_controller_queues()
                event = self.controller.step(action="GetReachablePositions")
            else:
                raise
        if not event.metadata["lastActionSuccess"]:
            return False

        self.reachable_positions = event.metadata["actionReturn"]
        # Guard against Unity returning None (broken controller / scene reset failure).
        # This used to crash the whole case on len(None); now we fail the reachability
        # build cleanly so the outer loop can skip / reset rather than explode.
        if self.reachable_positions is None:
            print("  [NAV] WARNING: GetReachablePositions returned None — Unity controller likely disconnected or scene state corrupted")
            self.reachable_positions = []
            return False
        self._map_built_at = now
        self.position_to_grid = {}
        self.grid_to_position = {}
        print(f"  [NAV] Reachable map: {len(self.reachable_positions)} positions, grid_size={self.grid_size}")

        for pos in self.reachable_positions:
            grid_x = round(pos["x"] / self.grid_size)
            grid_z = round(pos["z"] / self.grid_size)
            grid_key = (grid_x, grid_z)
            self.position_to_grid[(pos["x"], pos["z"])] = grid_key
            if grid_key not in self.grid_to_position:
                self.grid_to_position[grid_key] = (pos["x"], pos["y"], pos["z"])

        return True

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance heuristic for A*"""
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _get_neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring grid cells"""
        x, z = node
        neighbors = []
        for dx, dz in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_node = (x + dx, z + dz)
            if new_node in self.grid_to_position:
                neighbors.append(new_node)
        return neighbors

    def _reconstruct_path(
        self, came_from: Dict, current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Reconstruct path from A* search"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def find_path(
        self, start_pos: Tuple[float, float], end_pos: Tuple[float, float]
    ) -> Optional[List[Tuple[float, float]]]:
        """Find path between two positions using A*"""
        if not self.reachable_positions:
            self.build_reachable_map()

        start_grid = (
            round(start_pos[0] / self.grid_size),
            round(start_pos[1] / self.grid_size),
        )
        end_grid = (
            round(end_pos[0] / self.grid_size),
            round(end_pos[1] / self.grid_size),
        )

        if start_grid not in self.grid_to_position:
            start_grid = self._find_nearest_grid(start_pos)
        if end_grid not in self.grid_to_position:
            end_grid = self._find_nearest_grid(end_pos)

        if start_grid is None or end_grid is None:
            return None

        open_set = []
        heapq.heappush(open_set, (0, start_grid))
        came_from = {}
        g_score = {start_grid: 0}
        f_score = {start_grid: self._heuristic(start_grid, end_grid)}

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == end_grid:
                grid_path = self._reconstruct_path(came_from, current)
                return [self.grid_to_position[g] for g in grid_path]

            for neighbor in self._get_neighbors(current):
                tentative_g = g_score[current] + 1

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(
                        neighbor, end_grid
                    )
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))

        return None

    def _find_nearest_grid(self, pos: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        """Find nearest reachable grid cell to a position"""
        if not self.reachable_positions:
            return None

        min_dist = float("inf")
        nearest = None

        for p in self.reachable_positions:
            dist = math.sqrt((p["x"] - pos[0]) ** 2 + (p["z"] - pos[1]) ** 2)
            if dist < min_dist:
                min_dist = dist
                nearest = (
                    round(p["x"] / self.grid_size),
                    round(p["z"] / self.grid_size),
                )

        return nearest

    def _find_interaction_positions(self, object_id: str) -> List[dict]:
        """Find positions where the agent can interact with the object.
        
        Uses AI2-THOR's GetInteractablePoses (from ref-planner) to get exact
        positions with rotation and camera angles. Falls back to distance-based
        candidates if the API returns nothing.
        
        Returns list of dicts: {x, z, rotation, horizon} or {x, z} for fallback.
        """
        # Try GetInteractablePoses with progressively more camera horizons and both standings (crouch/stand)
        for horizons in [[0], [0, 30], [0, 30, -30], [0, 30, -30, 60]]:
            try:
                event = self.controller.step(
                    action="GetInteractablePoses",
                    objectId=object_id,
                    standings=[True, False],
                    horizons=horizons,
                )
                if not event.metadata["lastActionSuccess"]:
                    print(f"  [NAV] GetInteractablePoses failed for {object_id.split('|')[0]}: {event.metadata.get('errorMessage', 'unknown')}")
                    break  # No point trying more horizons if the action itself fails
                poses = event.metadata.get("actionReturn")
            except Exception as e:
                print(f"  [NAV] GetInteractablePoses error for {object_id.split('|')[0]}: {e}")
                poses = None
                break
            if poses:
                # Return as list of dicts with full pose info
                result = []
                for p in poses:
                    result.append({
                        "x": p["x"], "y": p["y"], "z": p["z"],
                        "rotation": p["rotation"], 
                        "horizon": p["horizon"],
                        "standing": p["standing"]
                    })
                return result

        # Fallback: distance-based candidates (old approach)
        target_pos = self._get_object_position(object_id)
        if target_pos is None:
            return []

        if not self.reachable_positions:
            self.build_reachable_map()

        INTERACTION_RANGE = 2.0

        candidates = []
        for p in self.reachable_positions:
            dist = math.sqrt((p["x"] - target_pos[0]) ** 2 + (p["z"] - target_pos[1]) ** 2)
            if dist <= INTERACTION_RANGE:
                candidates.append((dist, {"x": p["x"], "z": p["z"]}))

        if not candidates:
            all_positions = []
            for p in self.reachable_positions:
                dist = math.sqrt((p["x"] - target_pos[0]) ** 2 + (p["z"] - target_pos[1]) ** 2)
                all_positions.append((dist, {"x": p["x"], "z": p["z"]}))
            all_positions.sort(key=lambda c: c[0])
            return [pos for _, pos in all_positions[:5]]

        candidates.sort(key=lambda c: c[0])
        return [pos for _, pos in candidates]

    # Large surface receptacles — distance to center is misleading for these
    _LARGE_RECEPTACLES = {"CounterTop", "Shelf", "ShelvingUnit", "DiningTable",
                          "SideTable", "CoffeeTable", "Desk", "Bed", "Sofa", "Floor"}

    def _is_object_interactable(self, object_id: str) -> bool:
        """Check if an object is visible and within interaction distance."""
        objects = self.controller.last_event.metadata["objects"]
        for obj in objects:
            if obj["objectId"] == object_id:
                obj_type = obj.get("objectType", "")
                # Large receptacles: just need visibility + relaxed distance
                if obj_type in self._LARGE_RECEPTACLES:
                    return obj.get("visible", False) and obj.get("distance", float("inf")) <= 5.0
                return obj.get("visible", False) and obj.get("distance", float("inf")) <= 2.0
        return False

    def _center_object_in_frame(self, object_id: str):
        """
        After navigation, tilt the camera so the target object is vertically
        centered in the video feed. Uses the real 3D position delta to compute
        the ideal pitch.
        """
        try:
            # Get current agent state
            agent_meta = self.controller.last_event.metadata["agent"]
            agent_pos = agent_meta["position"]
            agent_height = agent_pos["y"] + (0.675 if agent_meta.get("isStanding", True) else 0.45)

            # Get object position from scene metadata
            objects = self.controller.last_event.metadata["objects"]
            target = next((o for o in objects if o["objectId"] == object_id), None)
            if not target:
                return

            obj_pos = target["position"]

            # Compute horizontal distance (XZ plane)
            dx = obj_pos["x"] - agent_pos["x"]
            dz = obj_pos["z"] - agent_pos["z"]
            horizontal_dist = math.sqrt(dx * dx + dz * dz)

            if horizontal_dist < 0.01:
                return

            # dy: positive = object below camera, negative = object above
            dy = agent_height - obj_pos["y"]

            # Target pitch: positive = look down, negative = look up (AI2-THOR horizon convention)
            target_horizon = math.degrees(math.atan2(dy, horizontal_dist))
            # Clamp to [-60, 60] AI2-THOR horizon range
            target_horizon = max(-60.0, min(60.0, target_horizon))

            current_horizon = agent_meta["cameraHorizon"]
            diff = target_horizon - current_horizon

            print(f"  [CAM] Centering {object_id.split('|')[0]} in frame: target horizon={target_horizon:.1f}°, current={current_horizon:.1f}°, diff={diff:.1f}°")

            if abs(diff) < 3.0:
                return  # Already close enough

            action = "LookDown" if diff > 0 else "LookUp"
            self.controller.step(action=action, degrees=abs(diff))

        except Exception as e:
            print(f"  [CAM] Could not center object in frame: {e}")

    def navigate_to_object(self, agent, object_id: str) -> Tuple[bool, str]:
        """Navigate to a position where we can interact with the target object.

        Uses GetInteractablePoses when available (exact position + rotation + horizon).
        Falls back to distance-based candidates otherwise.
        In cluttered scenes, uses direct teleport as last resort.
        """
        # Rebuild reachable map to account for any moved objects
        self.build_reachable_map()

        # Check if we can already interact without moving
        if self._is_object_interactable(object_id):
            print(f"  [NAV] Already within interaction range of {object_id.split('|')[0]}")
            self._center_object_in_frame(object_id)
            return True, f"Already near {object_id}"

        # Get candidate interaction positions
        candidates = self._find_interaction_positions(object_id)
        if not candidates:
            print(f"  [NAV] No interaction positions found for {object_id.split('|')[0]}")
            # Last resort: try proximity-based teleport even without interactable poses
            return self._proximity_teleport_fallback(agent, object_id)

        # Collect exact poses for teleport fallback (only from GetInteractablePoses results)
        exact_poses = [p for p in candidates if "rotation" in p]

        agent_pos = agent.get_agent_position()
        start_pos = (agent_pos["x"], agent_pos["z"])

        # Try each candidate position via pathfinding
        for attempt, pose in enumerate(candidates[:10]):
            target_pos = (pose["x"], pose["z"])
            path = self.find_path(start_pos, target_pos)
            if path is None:
                continue

            if attempt > 0:
                print(f"  [NAV] Trying alternate position #{attempt + 1}...")

            print(f"  [NAV] Pathfinding to interaction position near {object_id.split('|')[0]}...")
            print(f"  [NAV] Path: {len(path) - 1} steps")

            success = self._execute_path(agent, path, object_id)
            if success:
                # If we have exact pose from GetInteractablePoses, teleport to it for precision
                if "rotation" in pose:
                    print(f"  [NAV] Applying exact pose: pos=({pose['x']:.2f}, {pose['y']:.2f}, {pose['z']:.2f}), rot={pose['rotation']:.1f}, horiz={pose['horizon']:.1f}, stand={pose['standing']}")
                    self.controller.step(
                        action="Teleport",
                        position=dict(x=pose["x"], y=pose["y"], z=pose["z"]),
                        rotation=dict(x=0, y=pose["rotation"], z=0),
                        horizon=pose["horizon"],
                        standing=pose["standing"]
                    )

                # Verify object is interactable after positioning
                if self._is_object_interactable(object_id):
                    self._center_object_in_frame(object_id)
                    return True, f"Navigated to {object_id}"

                # Vertical scanning recovery step
                if self._vertical_recovery(object_id):
                    self._center_object_in_frame(object_id)
                    return True, f"Navigated to {object_id}"

                # Fallback to rotation recovery
                recovered = self._rotation_recovery(object_id)
                if recovered:
                    self._center_object_in_frame(object_id)
                    return True, f"Navigated to {object_id}"

            # Reset for next attempt
            agent_pos = agent.get_agent_position()
            start_pos = (agent_pos["x"], agent_pos["z"])

        # CLUTTERED SCENE FALLBACK: Walk as close as possible, then short teleport
        # When pathfinding to exact positions fails, walk to the nearest reachable
        # position first, then do a small teleport for final positioning
        if exact_poses:
            # First, walk as close as possible to the target object
            target_obj_pos = self._get_object_position(object_id)
            if target_obj_pos and self.reachable_positions:
                closest_reachable = min(
                    self.reachable_positions,
                    key=lambda p: math.sqrt((p["x"] - target_obj_pos[0])**2 + (p["z"] - target_obj_pos[1])**2)
                )
                walk_target = (closest_reachable["x"], closest_reachable["z"])
                walk_path = self.find_path(start_pos, walk_target)
                if walk_path and len(walk_path) > 1:
                    print(f"  [NAV] Walking to closest reachable point ({len(walk_path)-1} steps) before fine-positioning...")
                    self._execute_path(agent, walk_path, object_id)
                    # Update start_pos after walking
                    agent_pos = agent.get_agent_position()
                    start_pos = (agent_pos["x"], agent_pos["z"])

            # Now try short-range teleport to exact interactable poses
            max_teleport_attempts = min(50, len(exact_poses))
            print(f"  [NAV] Fine-positioning: trying {max_teleport_attempts}/{len(exact_poses)} interactable poses...")
            for i, pose in enumerate(exact_poses[:max_teleport_attempts]):
                event = self.controller.step(
                    action="Teleport",
                    position=dict(x=pose["x"], y=pose["y"], z=pose["z"]),
                    rotation=dict(x=0, y=pose["rotation"], z=0),
                    horizon=pose["horizon"],
                    standing=pose["standing"]
                )
                if event.metadata["lastActionSuccess"]:
                    if self._is_object_interactable(object_id):
                        print(f"  [NAV] Direct teleport #{i+1} succeeded for {object_id.split('|')[0]}")
                        self._center_object_in_frame(object_id)
                        return True, f"Teleported to {object_id}"
                    # Close enough — check with relaxed visibility (distance only)
                    for obj in self.controller.last_event.metadata["objects"]:
                        if obj["objectId"] == object_id and obj.get("distance", float("inf")) <= 1.5:
                            print(f"  [NAV] Direct teleport #{i+1}: object within reach (dist={obj['distance']:.2f}), proceeding")
                            self._center_object_in_frame(object_id)
                            return True, f"Teleported near {object_id}"

        # Last resort: proximity-based teleport (closest reachable positions with multiple rotations)
        print(f"  [NAV] Direct teleport exhausted, trying proximity fallback for {object_id.split('|')[0]}...")
        return self._proximity_teleport_fallback(agent, object_id)

    def _proximity_teleport_fallback(self, agent, object_id: str) -> Tuple[bool, str]:
        """Last resort: teleport to the closest reachable position to the object
        and try different rotations/horizons. For cluttered scenes where
        GetInteractablePoses returns nothing."""
        target_pos = self._get_object_position(object_id)
        if target_pos is None:
            return False, f"Object {object_id} not found"

        if not self.reachable_positions:
            self.build_reachable_map()

        # Find closest reachable positions
        positions_by_dist = []
        for p in self.reachable_positions:
            dist = math.sqrt((p["x"] - target_pos[0]) ** 2 + (p["z"] - target_pos[1]) ** 2)
            positions_by_dist.append((dist, p))
        positions_by_dist.sort(key=lambda x: x[0])

        num_positions = min(8, len(positions_by_dist))
        print(f"  [NAV] Proximity fallback: trying {num_positions} closest positions to {object_id.split('|')[0]}...")

        for dist, pos in positions_by_dist[:num_positions]:
            # Calculate rotation to face the object
            dx = target_pos[0] - pos["x"]
            dz = target_pos[1] - pos["z"]
            face_angle = math.degrees(math.atan2(dx, dz)) % 360

            # Try face angle first, then ±30° offsets (3 rotations instead of 7)
            for rot_offset in [0, 30, -30]:
                rot = (face_angle + rot_offset) % 360
                # 3 horizons instead of 5, standing only (halves attempts)
                for horizon in [0, 30, -30]:
                    event = self.controller.step(
                        action="Teleport",
                        position=pos,
                        rotation=dict(x=0, y=rot, z=0),
                        horizon=horizon,
                        standing=True
                    )
                    if event.metadata["lastActionSuccess"] and self._is_object_interactable(object_id):
                        print(f"  [NAV] Proximity fallback succeeded at dist={dist:.2f}m, rot={rot:.0f}, horizon={horizon}")
                        self._center_object_in_frame(object_id)
                        return True, f"Teleported near {object_id}"

        # Second pass with crouching if standing didn't work (fewer combos)
        for dist, pos in positions_by_dist[:3]:
            dx = target_pos[0] - pos["x"]
            dz = target_pos[1] - pos["z"]
            face_angle = math.degrees(math.atan2(dx, dz)) % 360
            for horizon in [0, 30, 60]:
                event = self.controller.step(
                    action="Teleport",
                    position=pos,
                    rotation=dict(x=0, y=face_angle, z=0),
                    horizon=horizon,
                    standing=False
                )
                if event.metadata["lastActionSuccess"] and self._is_object_interactable(object_id):
                    print(f"  [NAV] Proximity fallback (crouched) succeeded at dist={dist:.2f}m, horizon={horizon}")
                    self._center_object_in_frame(object_id)
                    return True, f"Teleported near {object_id}"

        # ABSOLUTE LAST RESORT: walk to the closest reachable position and report partial success
        # The executor's pickup will try forceAction=True (teleport-to-hand) as fallback
        if self.reachable_positions and target_pos:
            closest = min(
                self.reachable_positions,
                key=lambda p: math.sqrt((p["x"] - target_pos[0])**2 + (p["z"] - target_pos[1])**2)
            )
            closest_dist = math.sqrt((closest["x"] - target_pos[0])**2 + (closest["z"] - target_pos[1])**2)
            if closest_dist < 3.0:
                # Teleport to closest point facing the object
                dx = target_pos[0] - closest["x"]
                dz = target_pos[1] - closest["z"]
                face_angle = math.degrees(math.atan2(dx, dz)) % 360
                self.controller.step(
                    action="Teleport",
                    position=closest,
                    rotation=dict(x=0, y=face_angle, z=0),
                    horizon=0,
                    standing=True
                )
                print(f"  [NAV] Moved to closest point ({closest_dist:.2f}m from object) — executor will try teleport-to-hand")
                return True, f"Moved near {object_id} (may need force pickup)"

        return False, f"Object {object_id} not accessible (cluttered scene)"

    def _set_agent_rotation(self, agent, target_rotation: float):
        """Rotate the agent to face the exact target rotation."""
        agent_pos = agent.get_agent_position()
        current = agent_pos["rotation"] % 360
        target = target_rotation % 360
        diff = (target - current + 180) % 360 - 180

        if abs(diff) < 1:
            return

        rot_action = "RotateRight" if diff > 0 else "RotateLeft"
        steps = round(abs(diff) / 15)
        for _ in range(steps):
            self.controller.step(action=rot_action)

    def _set_camera_horizon(self, target_horizon: float):
        """Set the camera to the exact horizon angle."""
        current = self.controller.last_event.metadata["agent"]["cameraHorizon"]
        diff = target_horizon - current
        if abs(diff) < 1:
            return

        action = "LookDown" if diff > 0 else "LookUp"
        self.controller.step(action=action, degrees=abs(diff))

    def _set_agent_standing(self, standing: bool):
        """Set the agent to standing or crouching stance."""
        current_standing = self.controller.last_event.metadata["agent"].get("isStanding", True)
        if current_standing == standing:
            return
        
        if standing:
            self.controller.step(action="Stand")
        else:
            self.controller.step(action="Crouch")

    def _vertical_recovery(self, object_id: str) -> bool:
        """Try adjusting camera horizon and stance to find the target object."""
        print(f"  [RECOVERY] Target {object_id.split('|')[0]} not visible. Attempting vertical recovery...")
        
        # Strategy: Try different horizons for both standing and crouching
        horizons_to_try = [30, 60, 0, -30]
        standings_to_try = [True, False]
        
        agent_meta = self.controller.last_event.metadata["agent"]
        pos = agent_meta["position"]
        rot = agent_meta["rotation"]
        
        for standing in standings_to_try:
            for horizon in horizons_to_try:
                # Use Teleport for quick horizon/standing adjustment while keeping same position/rotation
                self.controller.step(
                    action="Teleport",
                    position=pos,
                    rotation=rot,
                    horizon=horizon,
                    standing=standing
                )
                if self._is_object_interactable(object_id):
                    print(f"  [RECOVERY] Found target at horizon {horizon} and standing={standing}!")
                    return True
        return False

    def _rotation_recovery(self, object_id: str) -> bool:
        """Try rotating in place to find the target object."""
        print(f"  [RECOVERY] Target {object_id.split('|')[0]} not visible. Attempting rotation recovery...")
        
        agent_meta = self.controller.last_event.metadata["agent"]
        pos = agent_meta["position"]
        horizon = agent_meta["cameraHorizon"]
        standing = agent_meta.get("isStanding", True)
        
        for angle in range(0, 360, 45):
            self.controller.step(
                action="Teleport",
                position=pos,
                rotation=dict(x=0, y=angle, z=0),
                horizon=horizon,
                standing=standing
            )
            if self._is_object_interactable(object_id):
                print(f"  [RECOVERY] Found target at rotation {angle}!")
                return True
        return False

    def _execute_path(self, agent, path, object_id: str) -> bool:
        """Walk a path using Teleport to each reachable waypoint.

        Uses Teleport instead of RotateRight+MoveAhead because:
        - With snapToGrid=False the agent can land between grid points
        - MoveAhead(moveMagnitude) must exactly match gridSize or the move fails
        - Teleport to known-reachable positions from GetReachablePositions is always valid

        Returns True if we reach the end or the target becomes interactable mid-path.
        """
        agent_pos = agent.get_agent_position()
        # Debug: show first waypoint to verify y-coordinate
        if len(path) > 1:
            wp0 = path[1]
            if len(wp0) == 3:
                print(f"  [NAV] First waypoint: ({wp0[0]:.2f}, y={wp0[1]:.3f}, {wp0[2]:.2f}), agent y={agent_pos['y']:.3f}")
            else:
                print(f"  [NAV] First waypoint: ({wp0[0]:.2f}, {wp0[1]:.2f}), agent y={agent_pos['y']:.3f} (no wp y!)")

        for i, waypoint in enumerate(path[1:], 1):
            # Unpack waypoint — now includes y from GetReachablePositions
            if len(waypoint) == 3:
                x, wp_y, z = waypoint
            else:
                x, z = waypoint
                wp_y = None

            current_x, current_z = agent_pos["x"], agent_pos["z"]

            # Calculate facing direction toward next waypoint
            dx = x - current_x
            dz = z - current_z
            face_angle = math.degrees(math.atan2(dx, dz)) % 360

            # Teleport to the next waypoint, facing the direction of travel
            agent_meta = self.controller.last_event.metadata["agent"]
            teleport_y = wp_y if wp_y is not None else agent_meta["position"]["y"]
            event = self.controller.step(
                action="Teleport",
                position=dict(x=x, y=teleport_y, z=z),
                rotation=dict(x=0, y=face_angle, z=0),
                horizon=agent_meta["cameraHorizon"],
                standing=agent_meta.get("isStanding", True),
            )

            if not event.metadata["lastActionSuccess"]:
                # Teleport to known-reachable position failed — try MoveAhead as fallback
                # First rotate toward waypoint
                rotation_diff = (face_angle - agent_pos["rotation"] + 180) % 360 - 180
                if abs(rotation_diff) > 5:
                    self.controller.step(
                        action="Teleport",
                        position=agent_meta["position"],
                        rotation=dict(x=0, y=face_angle, z=0),
                        horizon=agent_meta["cameraHorizon"],
                        standing=agent_meta.get("isStanding", True),
                    )
                # Try MoveAhead with exact grid step
                move_dist = math.sqrt(dx * dx + dz * dz)
                event = self.controller.step(action="MoveAhead", moveMagnitude=move_dist)

                if not event.metadata["lastActionSuccess"]:
                    # Check if target is already interactable
                    if self._is_object_interactable(object_id):
                        print(f"  [NAV] Movement blocked at step {i}, but target is already interactable.")
                        return True
                    return False  # This path failed, try next candidate

            agent_pos = agent.get_agent_position()
            print(f"  [NAV] Step {i}/{len(path)-1}: ({agent_pos['x']:.2f}, {agent_pos['z']:.2f})")

            # Early exit if target becomes interactable mid-path
            if self._is_object_interactable(object_id):
                print(f"  [NAV] Target reachable at step {i}/{len(path)-1}")
                return True

        print(f"  [NAV] Arrived near {object_id.split('|')[0]} after {len(path)-1} steps")
        return True

    def _get_object_position(self, object_id: str):
        """Get position of an object from metadata"""
        objects = self.controller.last_event.metadata["objects"]
        for obj in objects:
            if obj["objectId"] == object_id:
                pos = obj.get("position")
                if pos:
                    return (pos["x"], pos["z"])
        return None

    def get_steps_to_target(
        self, start_pos: Tuple[float, float], end_pos: Tuple[float, float]
    ) -> List[Dict]:
        """Get list of action steps to reach target (for LLM planning)"""
        path = self.find_path(start_pos, end_pos)
        if path is None:
            return []

        steps = []
        current_rotation = 0

        for i in range(len(path) - 1):
            wp = path[i]
            next_wp = path[i + 1]
            x, z = wp[0], wp[-1]  # works for both (x, z) and (x, y, z)
            next_x, next_z = next_wp[0], next_wp[-1]

            dx = next_x - x
            dz = next_z - z
            angle = math.degrees(math.atan2(dx, dz))

            rotation_diff = int((angle - current_rotation + 180) % 360 - 180)

            if abs(rotation_diff) > 15:
                rot_action = "RotateRight" if rotation_diff > 0 else "RotateLeft"
                rot_steps = abs(rotation_diff) // 15
                steps.append({"action": rot_action, "repeat": rot_steps})

            move_dist = math.sqrt(dx * dx + dz * dz)
            steps.append({"action": "MoveAhead", "moveMagnitude": round(move_dist, 4)})
            current_rotation = angle

        return steps
