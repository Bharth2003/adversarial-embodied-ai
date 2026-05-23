#!/usr/bin/env python3
"""
Standalone movement test for AI2-THOR Navigator.

Tests:
1. GetReachablePositions — how many? what y-values?
2. Teleport to a few reachable positions — do they succeed?
3. Navigate to a visible object using the Navigator
4. Navigate to a distant object using the Navigator

Run:  python test_movement.py
Requires: Unity build running on localhost:8200
"""

import sys
import math
import os
import types
import importlib.util

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ai2thor.controller import Controller

# Import navigator.py directly to skip __init__.py (which pulls in sounddevice)
_nav_spec = importlib.util.spec_from_file_location(
    "ai2thor_lab.navigator",
    os.path.join(os.path.dirname(__file__), "..", "..", "src", "ai2thor_lab", "navigator.py"),
)
_nav_mod = importlib.util.module_from_spec(_nav_spec)
_nav_spec.loader.exec_module(_nav_mod)
Navigator = _nav_mod.Navigator


def main():
    scene = "FloorPlan_Messy"
    print(f"=== Movement Test: {scene} ===\n")

    # 1. Connect to AI2-THOR
    print("[1] Connecting to AI2-THOR...")
    controller = Controller(
        scene=scene,
        width=800,
        height=600,
        gridSize=0.1,
        snapToGrid=False,
        rotateStepDegrees=15,
        start_unity=False,
        host="127.0.0.1",
        port=8200,
    )
    agent_meta = controller.last_event.metadata["agent"]
    agent_pos = agent_meta["position"]
    print(f"    Agent at: ({agent_pos['x']:.3f}, {agent_pos['y']:.3f}, {agent_pos['z']:.3f})")
    print(f"    Rotation: {agent_meta['rotation']['y']:.1f}°")
    print()

    # 2. GetReachablePositions
    print("[2] GetReachablePositions...")
    event = controller.step(action="GetReachablePositions")
    if not event.metadata["lastActionSuccess"]:
        print(f"    FAILED: {event.metadata.get('errorMessage', 'unknown')}")
        controller.stop()
        return

    positions = event.metadata["actionReturn"]
    print(f"    {len(positions)} reachable positions")

    # Check y-values
    y_values = set()
    for p in positions:
        y_values.add(round(p["y"], 4))
    print(f"    Unique y-values: {sorted(y_values)}")
    print(f"    Agent y: {agent_pos['y']:.4f}")
    if agent_pos["y"] not in [round(y, 4) for y in y_values]:
        y_diff = min(abs(agent_pos["y"] - y) for y in y_values)
        print(f"    ⚠ Agent y does NOT match any reachable y! Closest diff: {y_diff:.4f}")
    else:
        print(f"    ✓ Agent y matches reachable positions")
    print()

    # 3. Direct Teleport test — pick 5 random reachable positions
    print("[3] Direct Teleport test (5 reachable positions)...")
    import random
    random.seed(42)
    test_positions = random.sample(positions, min(5, len(positions)))

    for i, pos in enumerate(test_positions):
        event = controller.step(
            action="Teleport",
            position=dict(x=pos["x"], y=pos["y"], z=pos["z"]),
            rotation=dict(x=0, y=0, z=0),
            horizon=0,
            standing=True,
        )
        status = "✓" if event.metadata["lastActionSuccess"] else "✗"
        new_pos = controller.last_event.metadata["agent"]["position"]
        print(f"    {status} Teleport to ({pos['x']:.2f}, y={pos['y']:.3f}, {pos['z']:.2f}) → agent at ({new_pos['x']:.2f}, {new_pos['y']:.3f}, {new_pos['z']:.2f})")
        if not event.metadata["lastActionSuccess"]:
            print(f"      Error: {event.metadata.get('errorMessage', 'unknown')}")

    # Teleport with WRONG y to show the problem
    print("\n    Testing with WRONG y (agent's original y instead of reachable y)...")
    test_pos = test_positions[0]
    event = controller.step(
        action="Teleport",
        position=dict(x=test_pos["x"], y=agent_pos["y"], z=test_pos["z"]),
        rotation=dict(x=0, y=0, z=0),
        horizon=0,
        standing=True,
    )
    status = "✓" if event.metadata["lastActionSuccess"] else "✗"
    print(f"    {status} Teleport to ({test_pos['x']:.2f}, y={agent_pos['y']:.3f}[agent], {test_pos['z']:.2f})")
    if not event.metadata["lastActionSuccess"]:
        print(f"      Error: {event.metadata.get('errorMessage', 'unknown')}")
        print(f"      → This confirms the y-coordinate bug!")
    print()

    # 4. Reset agent to start position and test Navigator
    print("[4] Navigator path test...")
    controller.reset(scene=scene)
    agent_meta = controller.last_event.metadata["agent"]
    agent_pos = agent_meta["position"]
    print(f"    Agent reset to: ({agent_pos['x']:.3f}, {agent_pos['y']:.3f}, {agent_pos['z']:.3f})")

    nav = Navigator(controller)
    nav.build_reachable_map()
    print(f"    Grid size: {nav.grid_size}")
    print(f"    Grid map entries: {len(nav.grid_to_position)}")

    # Check that grid_to_position stores y
    sample_key = next(iter(nav.grid_to_position))
    sample_val = nav.grid_to_position[sample_key]
    print(f"    Sample grid_to_position: key={sample_key} → value={sample_val} (len={len(sample_val)})")
    if len(sample_val) == 3:
        print(f"    ✓ grid_to_position stores (x, y, z)")
    else:
        print(f"    ✗ grid_to_position stores only (x, z) — BUG NOT FIXED!")
    print()

    # 5. Pick an object and navigate to it
    print("[5] Navigate to an object...")
    objects = controller.last_event.metadata["objects"]

    # Find a pickupable object that's not too close
    target = None
    for obj in sorted(objects, key=lambda o: o.get("distance", 0), reverse=True):
        if obj.get("pickupable") and obj.get("distance", 0) > 1.0:
            target = obj
            break

    if not target:
        # Fall back to any visible object
        for obj in objects:
            if obj.get("visible") and obj.get("distance", 0) > 0.5:
                target = obj
                break

    if target:
        print(f"    Target: {target['objectType']} (id={target['objectId'][:30]}...)")
        print(f"    Distance: {target.get('distance', '?')}m, Visible: {target.get('visible')}")
        pos = target["position"]
        print(f"    Position: ({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f})")

        # Create a simple agent wrapper
        class SimpleAgent:
            def __init__(self, ctrl):
                self.controller = ctrl
                self.held_object = None
            def get_agent_position(self):
                meta = self.controller.last_event.metadata["agent"]
                return {
                    "x": meta["position"]["x"],
                    "y": meta["position"]["y"],
                    "z": meta["position"]["z"],
                    "rotation": meta["rotation"]["y"],
                    "horizon": meta["cameraHorizon"],
                    "standing": meta.get("isStanding", True),
                }

        agent = SimpleAgent(controller)
        success, msg = nav.navigate_to_object(agent, target["objectId"])
        print(f"    Result: success={success}, msg={msg}")

        # Check final position
        final = agent.get_agent_position()
        print(f"    Final agent position: ({final['x']:.2f}, {final['y']:.3f}, {final['z']:.2f})")
    else:
        print("    No suitable target found!")

    print("\n=== Test Complete ===")
    controller.stop()


if __name__ == "__main__":
    main()
