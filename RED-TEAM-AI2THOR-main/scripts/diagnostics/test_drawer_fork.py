"""
Diagnostic script: Test whether objects inside drawers become visible/accessible
after opening the drawer in AI2-THOR FloorPlan_Messy.

Run with: uv run python test_drawer_fork.py

Goes slow so you can watch what happens in the Unity window.
"""
import time
from ai2thor.controller import Controller

PAUSE = 2.0  # seconds between actions — adjust to go faster/slower

def pause(msg=""):
    if msg:
        print(f"\n{'='*60}")
        print(f"  {msg}")
        print(f"{'='*60}")
    time.sleep(PAUSE)


def find_objects_by_type(objects, obj_type):
    """Find all objects of a given type."""
    return [o for o in objects if o["objectType"] == obj_type]


def find_forks_in_drawers(objects):
    """Find forks that are inside drawers (parentReceptacles contains a Drawer)."""
    forks = find_objects_by_type(objects, "Fork")
    in_drawer = []
    for f in forks:
        parents = f.get("parentReceptacles") or []
        parent_types = [p.split("|")[0] for p in parents]
        if "Drawer" in parent_types:
            in_drawer.append(f)
    return in_drawer


def print_fork_status(objects, label=""):
    """Print visibility and position status of all forks."""
    forks = find_objects_by_type(objects, "Fork")
    print(f"\n--- Fork Status {label} ({len(forks)} total) ---")
    for f in forks:
        oid = f["objectId"]
        short = oid.split("|")[0] + "|" + "|".join(f"{float(x):+.2f}" for x in oid.split("|")[1:])
        parents = f.get("parentReceptacles") or []
        parent_types = [p.split("|")[0] for p in parents]
        vis = f.get("visible", False)
        dist = f.get("distance", -1)
        pickup = f.get("pickupable", False)
        pos = f.get("position", {})
        print(f"  {short}")
        print(f"    visible={vis}  distance={dist:.2f}m  pickupable={pickup}")
        print(f"    parents={parent_types}  pos=({pos.get('x',0):.2f}, {pos.get('y',0):.2f}, {pos.get('z',0):.2f})")
    return forks


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
    print(f"Scene loaded: {len(objects)} objects")

    # Find all drawers
    drawers = find_objects_by_type(objects, "Drawer")
    print(f"\nFound {len(drawers)} drawers:")
    for d in drawers:
        oid = d["objectId"]
        is_open = d.get("isOpen", False)
        pos = d.get("position", {})
        children = d.get("receptacleObjectIds", [])
        child_types = [c.split("|")[0] for c in children]
        print(f"  {oid.split('|')[0]}|...  open={is_open}  pos=({pos.get('x',0):.2f}, {pos.get('y',0):.2f}, {pos.get('z',0):.2f})")
        print(f"    contains: {child_types}")

    # Find forks in drawers
    forks_in_drawers = find_forks_in_drawers(objects)
    if not forks_in_drawers:
        print("\nNo forks found inside drawers! Checking all fork locations...")
        print_fork_status(objects, "INITIAL")
        controller.stop()
        return

    target_fork = forks_in_drawers[0]
    fork_id = target_fork["objectId"]
    fork_parents = target_fork.get("parentReceptacles", [])
    drawer_id = None
    for p in fork_parents:
        if p.split("|")[0] == "Drawer":
            drawer_id = p
            break

    print(f"\n>>> Target fork: {fork_id}")
    print(f">>> Inside drawer: {drawer_id}")

    # ── STEP 1: Check fork visibility BEFORE opening drawer ──
    pause("STEP 1: Checking fork visibility BEFORE opening drawer")
    print_fork_status(objects, "BEFORE OPEN")

    fork_obj = next((o for o in objects if o["objectId"] == fork_id), None)
    if fork_obj:
        print(f"\n  Target fork visible BEFORE open? {fork_obj.get('visible', False)}")

    # ── STEP 2: Navigate near the drawer ──
    pause("STEP 2: Navigating near the drawer")
    drawer_obj = next((o for o in objects if o["objectId"] == drawer_id), None)
    if drawer_obj:
        dpos = drawer_obj["position"]
        # Teleport agent near the drawer
        event = controller.step(
            action="Teleport",
            position={"x": dpos["x"] + 0.6, "y": 0.9, "z": dpos["z"]},
            rotation={"x": 0, "y": 270, "z": 0},  # Face the drawer
            horizon=30,
            standing=True,
        )
        if event.metadata["lastActionSuccess"]:
            print(f"  Teleported near drawer at ({dpos['x']+0.6:.2f}, 0.9, {dpos['z']:.2f})")
        else:
            print(f"  Teleport failed: {event.metadata.get('errorMessage', 'unknown')}")
            # Try slightly different position
            event = controller.step(
                action="Teleport",
                position={"x": dpos["x"] + 0.8, "y": 0.9, "z": dpos["z"]},
                rotation={"x": 0, "y": 270, "z": 0},
                horizon=30,
                standing=True,
            )
            print(f"  Retry teleport: {'OK' if event.metadata['lastActionSuccess'] else 'FAILED'}")

    # Check visibility after moving near
    objects = controller.last_event.metadata["objects"]
    fork_obj = next((o for o in objects if o["objectId"] == fork_id), None)
    print(f"  Fork visible after moving near (drawer still closed)? {fork_obj.get('visible', False) if fork_obj else 'NOT FOUND'}")

    # ── STEP 3: Open the drawer ──
    pause("STEP 3: Opening the drawer")
    event = controller.step(action="OpenObject", objectId=drawer_id)
    if event.metadata["lastActionSuccess"]:
        print(f"  Drawer opened successfully!")
    else:
        print(f"  Open failed: {event.metadata.get('errorMessage', 'unknown')}")
        # Try with forceAction
        event = controller.step(action="OpenObject", objectId=drawer_id, forceAction=True)
        print(f"  Force open: {'OK' if event.metadata['lastActionSuccess'] else 'FAILED'}")

    time.sleep(1.0)  # Let physics settle

    # ── STEP 4: Check fork visibility AFTER opening drawer ──
    pause("STEP 4: Checking fork visibility AFTER opening drawer")
    objects = controller.last_event.metadata["objects"]
    print_fork_status(objects, "AFTER OPEN")

    fork_obj = next((o for o in objects if o["objectId"] == fork_id), None)
    if fork_obj:
        print(f"\n  >>> Target fork visible AFTER open? {fork_obj.get('visible', False)}")
        print(f"  >>> Distance: {fork_obj.get('distance', -1):.2f}m")
        print(f"  >>> Pickupable: {fork_obj.get('pickupable', False)}")
        print(f"  >>> Position: {fork_obj.get('position', {})}")

    # ── STEP 5: Try to look down at the drawer contents ──
    pause("STEP 5: Looking down at drawer contents")
    event = controller.step(action="LookDown", degrees=15)
    time.sleep(0.5)

    objects = controller.last_event.metadata["objects"]
    fork_obj = next((o for o in objects if o["objectId"] == fork_id), None)
    print(f"  Fork visible after looking down? {fork_obj.get('visible', False) if fork_obj else 'NOT FOUND'}")

    # Try more looking down
    event = controller.step(action="LookDown", degrees=15)
    time.sleep(0.5)
    objects = controller.last_event.metadata["objects"]
    fork_obj = next((o for o in objects if o["objectId"] == fork_id), None)
    print(f"  Fork visible after looking further down? {fork_obj.get('visible', False) if fork_obj else 'NOT FOUND'}")

    # ── STEP 6: Try to pick up the fork (even if not visible) ──
    pause("STEP 6: Attempting to pick up the fork")
    event = controller.step(action="PickupObject", objectId=fork_id)
    if event.metadata["lastActionSuccess"]:
        print(f"  PICKED UP fork successfully! (without forceAction)")
    else:
        print(f"  Normal pickup failed: {event.metadata.get('errorMessage', 'unknown')}")

        # Try with forceAction
        event = controller.step(action="PickupObject", objectId=fork_id, forceAction=True)
        if event.metadata["lastActionSuccess"]:
            print(f"  PICKED UP fork with forceAction!")
        else:
            print(f"  Force pickup also failed: {event.metadata.get('errorMessage', 'unknown')}")

    # ── STEP 7: Check what the agent is holding ──
    pause("STEP 7: Checking what agent is holding")
    objects = controller.last_event.metadata["objects"]
    held = [o for o in objects if o.get("isPickedUp", False)]
    if held:
        print(f"  Agent is holding: {[h['objectId'].split('|')[0] for h in held]}")
    else:
        print(f"  Agent is NOT holding anything")

    # ── STEP 8: If we're holding the fork, try putting it somewhere ──
    if held:
        pause("STEP 8: Placing fork on CounterTop")
        counters = find_objects_by_type(objects, "CounterTop")
        if counters:
            counter_id = counters[0]["objectId"]
            event = controller.step(
                action="PutObject",
                objectId=counter_id,
                forceAction=True,
            )
            if event.metadata["lastActionSuccess"]:
                print(f"  Placed fork on counter!")
            else:
                print(f"  Put failed: {event.metadata.get('errorMessage', 'unknown')}")

    # ── SUMMARY ──
    pause("DIAGNOSTIC COMPLETE")
    print("\nKey findings:")
    print(f"  - Fork was in drawer: {drawer_id}")

    # Final check
    objects = controller.last_event.metadata["objects"]
    fork_obj = next((o for o in objects if o["objectId"] == fork_id), None)
    if fork_obj:
        parents = fork_obj.get("parentReceptacles") or []
        print(f"  - Fork final location: {[p.split('|')[0] for p in parents]}")
        print(f"  - Fork visible: {fork_obj.get('visible', False)}")

    print("\nDone! Stopping controller...")
    controller.stop()


if __name__ == "__main__":
    main()
