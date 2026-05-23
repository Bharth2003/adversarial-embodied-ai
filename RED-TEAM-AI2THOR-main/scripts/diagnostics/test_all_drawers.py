"""
Diagnostic script: Open ALL drawers one by one so you can see in Unity
whether objects are properly placed or clustered inside.

Run with: uv run python test_all_drawers.py

Opens each drawer slowly so you can watch in the Unity window.
After opening all drawers, tries to pick up one item from each.
"""
import time
from ai2thor.controller import Controller

PAUSE = 3.0  # seconds between actions — adjust to see more/less

def pause(msg=""):
    if msg:
        print(f"\n{'='*60}")
        print(f"  {msg}")
        print(f"{'='*60}")
    time.sleep(PAUSE)


def main():
    print("Initializing AI2-THOR (start_unity=False, port 8200)...")
    print("Make sure Unity is running and connected!\n")

    controller = Controller(
        scene="FloorPlan_Messy",
        width=800, height=600,
        gridSize=0.1, snapToGrid=False,
        rotateStepDegrees=15,
        start_unity=False,
        host="127.0.0.1", port=8200,
    )

    objects = controller.last_event.metadata["objects"]
    print(f"Scene loaded: {len(objects)} objects\n")

    # Find all drawers, sorted by position for clarity
    drawers = [o for o in objects if o["objectType"] == "Drawer"]
    drawers.sort(key=lambda d: (d["position"]["x"], -d["position"]["y"], d["position"]["z"]))

    print(f"Found {len(drawers)} drawers:\n")
    for i, d in enumerate(drawers):
        pos = d["position"]
        children = d.get("receptacleObjectIds", [])
        child_types = [c.split("|")[0] for c in children]
        height = "TOP" if pos["y"] > 0.75 else "MID" if pos["y"] > 0.45 else "BOT"
        side = "LEFT" if pos["x"] < 0 else "RIGHT"
        print(f"  Drawer {i+1}: {side} wall, {height} (y={pos['y']:.2f})")
        print(f"    Position: ({pos['x']:.2f}, {pos['y']:.2f}, {pos['z']:.2f})")
        print(f"    Contains {len(children)} items: {child_types}")
        print()

    # ── Open each drawer one by one ──
    for i, drawer in enumerate(drawers):
        drawer_id = drawer["objectId"]
        pos = drawer["position"]
        children = drawer.get("receptacleObjectIds", [])
        child_types = [c.split("|")[0] for c in children]
        height = "TOP" if pos["y"] > 0.75 else "MID" if pos["y"] > 0.45 else "BOT"
        side = "LEFT" if pos["x"] < 0 else "RIGHT"

        pause(f"DRAWER {i+1}/{len(drawers)}: {side} {height} — {len(children)} items: {child_types}")

        # Teleport near the drawer and face it
        # Figure out which direction to face based on drawer position
        if pos["x"] < -1.0:
            # Left wall drawers — approach from the right, face left (270°)
            tp_x = pos["x"] + 0.7
            tp_z = pos["z"]
            rotation = 270
        else:
            # Right wall drawers — approach from the left, face right (90°)
            tp_x = pos["x"] - 0.7
            tp_z = pos["z"]
            rotation = 90

        # Adjust horizon based on drawer height
        if pos["y"] > 0.75:
            horizon = 15  # Look slightly down for top drawers
        elif pos["y"] > 0.45:
            horizon = 30  # Look more down for mid drawers
        else:
            horizon = 45  # Look quite down for bottom drawers

        event = controller.step(
            action="Teleport",
            position={"x": tp_x, "y": 0.9, "z": tp_z},
            rotation={"x": 0, "y": rotation, "z": 0},
            horizon=horizon,
            standing=True,
        )

        if not event.metadata["lastActionSuccess"]:
            # Try slightly different position
            event = controller.step(
                action="Teleport",
                position={"x": tp_x + 0.1, "y": 0.9, "z": tp_z + 0.1},
                rotation={"x": 0, "y": rotation, "z": 0},
                horizon=horizon,
                standing=True,
            )
            if not event.metadata["lastActionSuccess"]:
                print(f"  Could not teleport near drawer: {event.metadata.get('errorMessage', '')[:80]}")
                continue

        print(f"  Teleported to ({tp_x:.2f}, 0.9, {tp_z:.2f}), facing {rotation}°, horizon {horizon}°")

        # Check visibility of items BEFORE opening
        objects = controller.last_event.metadata["objects"]
        for child_id in children:
            child_obj = next((o for o in objects if o["objectId"] == child_id), None)
            if child_obj:
                ctype = child_id.split("|")[0]
                vis = child_obj.get("visible", False)
                dist = child_obj.get("distance", -1)
                cpos = child_obj.get("position", {})
                print(f"  BEFORE open: {ctype:15s} visible={vis}  dist={dist:.2f}m  pos=({cpos.get('x',0):.2f}, {cpos.get('y',0):.2f}, {cpos.get('z',0):.2f})")

        time.sleep(1.0)

        # Open the drawer
        print(f"\n  >>> Opening drawer...")
        event = controller.step(action="OpenObject", objectId=drawer_id, forceAction=True)
        if event.metadata["lastActionSuccess"]:
            print(f"  >>> Drawer OPENED")
        else:
            print(f"  >>> Open FAILED: {event.metadata.get('errorMessage', '')[:80]}")
            continue

        time.sleep(1.5)  # Let physics settle and give time to watch

        # Check visibility AFTER opening
        objects = controller.last_event.metadata["objects"]
        visible_count = 0
        for child_id in children:
            child_obj = next((o for o in objects if o["objectId"] == child_id), None)
            if child_obj:
                ctype = child_id.split("|")[0]
                vis = child_obj.get("visible", False)
                dist = child_obj.get("distance", -1)
                cpos = child_obj.get("position", {})
                status = "VISIBLE" if vis else "HIDDEN"
                if vis:
                    visible_count += 1
                print(f"  AFTER open:  {ctype:15s} {status:7s}  dist={dist:.2f}m  pos=({cpos.get('x',0):.2f}, {cpos.get('y',0):.2f}, {cpos.get('z',0):.2f})")

        print(f"\n  VISIBILITY: {visible_count}/{len(children)} items visible after opening")

        # Try forceAction pickup on first item to verify it works
        if children:
            test_id = children[0]
            test_type = test_id.split("|")[0]
            print(f"\n  Testing pickup of {test_type}...")
            event = controller.step(action="PickupObject", objectId=test_id)
            if event.metadata["lastActionSuccess"]:
                print(f"    Normal pickup: SUCCESS")
            else:
                print(f"    Normal pickup: FAILED ({event.metadata.get('errorMessage', '')[:60]})")
                event = controller.step(action="PickupObject", objectId=test_id, forceAction=True)
                if event.metadata["lastActionSuccess"]:
                    print(f"    forceAction pickup: SUCCESS")
                else:
                    print(f"    forceAction pickup: FAILED ({event.metadata.get('errorMessage', '')[:60]})")

            # Drop it back
            if event.metadata["lastActionSuccess"]:
                controller.step(action="DropHandObject", forceAction=True)
                print(f"    (dropped back)")

        time.sleep(1.0)

    # ── FINAL: Open ALL drawers at once so user can see everything ──
    pause("OPENING ALL DRAWERS AT ONCE — look at Unity!")

    # Reset scene first
    controller.step(action="Done")
    objects = controller.last_event.metadata["objects"]
    drawers = [o for o in objects if o["objectType"] == "Drawer"]

    for d in drawers:
        controller.step(action="OpenObject", objectId=d["objectId"], forceAction=True)
        time.sleep(0.3)

    print("  All drawers opened! Check Unity to see object placement.")
    print("  Look for:")
    print("    - Objects clumped together / overlapping")
    print("    - Objects that fell through the drawer bottom")
    print("    - Objects that are below the drawer (wrong Y position)")
    print("    - Objects that are hidden behind the drawer face")

    # Teleport to a good overview position
    controller.step(
        action="Teleport",
        position={"x": -0.5, "y": 0.9, "z": -0.5},
        rotation={"x": 0, "y": 270, "z": 0},
        horizon=20,
        standing=True,
    )

    time.sleep(5.0)

    # Final summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    objects = controller.last_event.metadata["objects"]

    total_in_drawers = 0
    total_visible = 0
    for d in drawers:
        children = d.get("receptacleObjectIds", [])
        for child_id in children:
            total_in_drawers += 1
            child_obj = next((o for o in objects if o["objectId"] == child_id), None)
            if child_obj and child_obj.get("visible", False):
                total_visible += 1

    print(f"  Total objects in drawers: {total_in_drawers}")
    print(f"  Visible after ALL drawers open: {total_visible}")
    print(f"  Hidden (need forceAction): {total_in_drawers - total_visible}")
    print(f"\n  Conclusion: {'Most items need forceAction pickup' if total_visible < total_in_drawers / 2 else 'Most items are visible after opening'}")

    input("\nPress Enter to close...")
    controller.stop()


if __name__ == "__main__":
    main()
