"""
AI2-THOR Diagnostic Test Script
Tests which objects can be picked up, placed, opened, toggled, filled, sliced, broken, etc.
Run this to understand scene capabilities before benchmarking.

Usage:
    python diagnostic_test.py [--scene FloorPlan1]
"""
import sys
import time
import json
from collections import defaultdict

# Default scene
SCENE = "FloorPlan1"
if "--scene" in sys.argv:
    idx = sys.argv.index("--scene")
    if idx + 1 < len(sys.argv):
        SCENE = sys.argv[idx + 1]

print(f"{'='*70}")
print(f"  AI2-THOR DIAGNOSTIC TEST — Scene: {SCENE}")
print(f"{'='*70}\n")

from ai2thor.controller import Controller

# Connect to existing Unity server (same setup as Agent class)
controller = Controller(
    scene=SCENE,
    width=800,
    height=600,
    gridSize=0.1,
    snapToGrid=False,
    rotateStepDegrees=15,
    start_unity=False,
    host="127.0.0.1",
    port=8200,
)
time.sleep(1)

objects = controller.last_event.metadata["objects"]
print(f"Total objects in scene: {len(objects)}\n")

# ── 1. OBJECT INVENTORY ──
print(f"{'─'*70}")
print("1. OBJECT INVENTORY")
print(f"{'─'*70}")

by_type = defaultdict(list)
for obj in objects:
    by_type[obj["objectType"]].append(obj)

for t in sorted(by_type.keys()):
    objs = by_type[t]
    props = []
    o = objs[0]
    if o.get("pickupable"): props.append("pickupable")
    if o.get("receptacle"): props.append("receptacle")
    if o.get("openable"): props.append("openable")
    if o.get("toggleable"): props.append("toggleable")
    if o.get("breakable"): props.append("breakable")
    if o.get("sliceable"): props.append("sliceable")
    if o.get("canFillWithLiquid"): props.append("fillable")
    if o.get("cookable"): props.append("cookable")
    if o.get("dirtyable"): props.append("dirtyable")
    print(f"  {t:25s} x{len(objs):2d}  [{', '.join(props) if props else 'static'}]")

# ── 2. PICKUP TEST ──
print(f"\n{'─'*70}")
print("2. PICKUP TEST — Can the agent pick up each pickupable object?")
print(f"{'─'*70}")

pickupable = [o for o in objects if o.get("pickupable")]
pickup_results = {"success": [], "fail": []}

for obj in pickupable:
    oid = obj["objectId"]
    otype = obj["objectType"]

    # Reset scene to clean state for each test
    controller.reset(scene=SCENE)
    time.sleep(0.2)

    # Try to teleport near the object and pick it up
    event = controller.step(
        action="PickupObject",
        objectId=oid,
        forceAction=True,
    )
    if event.metadata["lastActionSuccess"]:
        pickup_results["success"].append(otype)
        # Drop it
        controller.step(action="DropHandObject", forceAction=True)
    else:
        err = event.metadata.get("errorMessage", "unknown")
        pickup_results["fail"].append((otype, err[:60]))

print(f"  ✓ Can pick up ({len(pickup_results['success'])}): {', '.join(sorted(set(pickup_results['success'])))}")
print(f"  ✗ Cannot pick up ({len(pickup_results['fail'])}):")
for name, err in pickup_results["fail"]:
    print(f"      {name}: {err}")

# ── 3. TOGGLE TEST ──
print(f"\n{'─'*70}")
print("3. TOGGLE TEST — Can each toggleable object be turned on/off?")
print(f"{'─'*70}")

controller.reset(scene=SCENE)
time.sleep(0.2)
objects = controller.last_event.metadata["objects"]
toggleable = [o for o in objects if o.get("toggleable")]
toggle_results = {"on_success": [], "on_fail": [], "off_success": [], "off_fail": []}

for obj in toggleable:
    oid = obj["objectId"]
    otype = obj["objectType"]

    # Try toggle ON
    event = controller.step(action="ToggleObjectOn", objectId=oid, forceAction=True)
    if event.metadata["lastActionSuccess"]:
        toggle_results["on_success"].append(otype)
        # Try toggle OFF
        event2 = controller.step(action="ToggleObjectOff", objectId=oid, forceAction=True)
        if event2.metadata["lastActionSuccess"]:
            toggle_results["off_success"].append(otype)
        else:
            toggle_results["off_fail"].append((otype, event2.metadata.get("errorMessage", "")[:60]))
    else:
        err = event.metadata.get("errorMessage", "unknown")
        toggle_results["on_fail"].append((otype, err[:60]))

print(f"  ✓ Toggle ON works ({len(toggle_results['on_success'])}): {', '.join(sorted(set(toggle_results['on_success'])))}")
print(f"  ✗ Toggle ON fails ({len(toggle_results['on_fail'])}):")
for name, err in toggle_results["on_fail"]:
    print(f"      {name}: {err}")
print(f"  ✓ Toggle OFF works ({len(toggle_results['off_success'])}): {', '.join(sorted(set(toggle_results['off_success'])))}")

# ── 4. OPEN/CLOSE TEST ──
print(f"\n{'─'*70}")
print("4. OPEN/CLOSE TEST — Can each openable object be opened/closed?")
print(f"{'─'*70}")

controller.reset(scene=SCENE)
time.sleep(0.2)
objects = controller.last_event.metadata["objects"]
openable = [o for o in objects if o.get("openable")]
open_results = {"open_ok": [], "open_fail": [], "close_ok": [], "close_fail": []}

for obj in openable:
    oid = obj["objectId"]
    otype = obj["objectType"]

    event = controller.step(action="OpenObject", objectId=oid, forceAction=True)
    if event.metadata["lastActionSuccess"]:
        open_results["open_ok"].append(otype)
        event2 = controller.step(action="CloseObject", objectId=oid, forceAction=True)
        if event2.metadata["lastActionSuccess"]:
            open_results["close_ok"].append(otype)
        else:
            open_results["close_fail"].append((otype, event2.metadata.get("errorMessage", "")[:60]))
    else:
        open_results["open_fail"].append((otype, event.metadata.get("errorMessage", "")[:60]))

print(f"  ✓ Open works ({len(open_results['open_ok'])}): {', '.join(sorted(set(open_results['open_ok'])))}")
print(f"  ✗ Open fails ({len(open_results['open_fail'])}):")
for name, err in open_results["open_fail"]:
    print(f"      {name}: {err}")

# ── 5. FILL WITH LIQUID TEST ──
print(f"\n{'─'*70}")
print("5. FILL LIQUID TEST — Which objects can be filled?")
print(f"{'─'*70}")

controller.reset(scene=SCENE)
time.sleep(0.2)
objects = controller.last_event.metadata["objects"]
fillable = [o for o in objects if o.get("canFillWithLiquid")]
fill_results = {"success": [], "fail": []}

for obj in fillable:
    oid = obj["objectId"]
    otype = obj["objectType"]

    # Must be held to fill
    controller.step(action="PickupObject", objectId=oid, forceAction=True)
    for liquid in ["water", "coffee", "wine"]:
        event = controller.step(
            action="FillObjectWithLiquid",
            objectId=oid,
            fillLiquid=liquid,
            forceAction=True,
        )
        if event.metadata["lastActionSuccess"]:
            fill_results["success"].append(f"{otype}+{liquid}")
        else:
            fill_results["fail"].append(f"{otype}+{liquid}")
        # Empty it for next test
        controller.step(action="EmptyLiquidFromObject", objectId=oid, forceAction=True)

    controller.step(action="DropHandObject", forceAction=True)

print(f"  ✓ Fill works ({len(fill_results['success'])}): {', '.join(sorted(set(fill_results['success'])))}")
print(f"  ✗ Fill fails ({len(fill_results['fail'])}): {', '.join(sorted(set(fill_results['fail'])))}")

# ── 6. PLACEMENT TEST — Which receptacles accept which items? ──
print(f"\n{'─'*70}")
print("6. PLACEMENT TEST — Key object+receptacle combos from our dataset")
print(f"{'─'*70}")

# Test specific combos that appear in our attack/safe datasets
test_combos = [
    ("Egg", "Microwave"),
    ("Fork", "Microwave"),
    ("Knife", "Microwave"),
    ("Spoon", "Microwave"),
    ("CreditCard", "Microwave"),
    ("Book", "Microwave"),
    ("Potato", "StoveBurner"),
    ("Egg", "StoveBurner"),
    ("Knife", "StoveBurner"),
    ("Fork", "StoveBurner"),
    ("Mug", "StoveBurner"),
    ("Pan", "StoveBurner"),
    ("Pot", "StoveBurner"),
    ("Bread", "Toaster"),
    ("DishSponge", "Cup"),
    ("DishSponge", "Mug"),
    ("DishSponge", "Toaster"),
    ("Potato", "Pot"),
    ("Egg", "Bowl"),
    ("Apple", "Microwave"),
    ("Laptop", "SinkBasin"),
    ("Laptop", "GarbageCan"),
    ("CreditCard", "GarbageCan"),
]

controller.reset(scene=SCENE)
time.sleep(0.2)
objects = controller.last_event.metadata["objects"]

# Build lookup
obj_by_type = {}
for o in objects:
    t = o["objectType"]
    if t not in obj_by_type:
        obj_by_type[t] = o

placement_results = []

for item_type, recep_type in test_combos:
    controller.reset(scene=SCENE)
    time.sleep(0.1)
    objects = controller.last_event.metadata["objects"]

    item = None
    recep = None
    for o in objects:
        if o["objectType"] == item_type and item is None:
            item = o
        if o["objectType"] == recep_type and recep is None:
            recep = o

    if not item:
        placement_results.append((item_type, recep_type, "SKIP", f"{item_type} not in scene"))
        continue
    if not recep:
        placement_results.append((item_type, recep_type, "SKIP", f"{recep_type} not in scene"))
        continue

    # Open receptacle if needed
    if recep.get("openable"):
        controller.step(action="OpenObject", objectId=recep["objectId"], forceAction=True)

    # Pick up item
    event = controller.step(action="PickupObject", objectId=item["objectId"], forceAction=True)
    if not event.metadata["lastActionSuccess"]:
        placement_results.append((item_type, recep_type, "FAIL", f"Can't pick up {item_type}"))
        continue

    # Place on receptacle
    event = controller.step(
        action="PutObject",
        objectId=recep["objectId"],
        forceAction=True,
    )
    if event.metadata["lastActionSuccess"]:
        placement_results.append((item_type, recep_type, "OK", "Placed successfully"))
    else:
        err = event.metadata.get("errorMessage", "unknown")
        placement_results.append((item_type, recep_type, "FAIL", err[:80]))

    controller.step(action="DropHandObject", forceAction=True)

for item, recep, status, detail in placement_results:
    icon = "✓" if status == "OK" else ("⊘" if status == "SKIP" else "✗")
    color = ""
    print(f"  {icon} {item:20s} → {recep:15s}  {status:5s}  {detail}")

# ── 7. COOKING TEST ──
print(f"\n{'─'*70}")
print("7. COOK TEST — Which items can be cooked (on stove or microwave)?")
print(f"{'─'*70}")

controller.reset(scene=SCENE)
time.sleep(0.2)
objects = controller.last_event.metadata["objects"]
cookable = [o for o in objects if o.get("cookable")]
print(f"  Cookable objects: {', '.join(sorted(set(o['objectType'] for o in cookable)))}")

# Check which are already cooked
for obj in cookable:
    cooked = obj.get("isCooked", False)
    print(f"    {obj['objectType']:20s} isCooked={cooked}")

# ── 8. BREAK TEST ──
print(f"\n{'─'*70}")
print("8. BREAK TEST — Which items can be broken?")
print(f"{'─'*70}")

breakable = [o for o in objects if o.get("breakable")]
break_results = {"success": [], "fail": []}

for obj in breakable:
    controller.reset(scene=SCENE)
    time.sleep(0.1)
    oid = obj["objectId"]
    otype = obj["objectType"]

    # Pick up and throw (breaking method)
    controller.step(action="PickupObject", objectId=oid, forceAction=True)
    event = controller.step(
        action="ThrowObject",
        moveMagnitude=150,
        forceAction=True,
    )
    # Check if broken
    time.sleep(0.3)
    objects_after = controller.last_event.metadata["objects"]
    broken = any(o.get("isBroken") and o["objectType"] == otype for o in objects_after)
    if broken:
        break_results["success"].append(otype)
    else:
        break_results["fail"].append(otype)

print(f"  ✓ Can break ({len(break_results['success'])}): {', '.join(sorted(set(break_results['success'])))}")
print(f"  ? Didn't break ({len(break_results['fail'])}): {', '.join(sorted(set(break_results['fail'])))}")

# ── 9. SLICE TEST ──
print(f"\n{'─'*70}")
print("9. SLICE TEST — Which items can be sliced?")
print(f"{'─'*70}")

controller.reset(scene=SCENE)
time.sleep(0.2)
objects = controller.last_event.metadata["objects"]
sliceable = [o for o in objects if o.get("sliceable")]
slice_results = {"success": [], "fail": []}

for obj in sliceable:
    controller.reset(scene=SCENE)
    time.sleep(0.1)
    oid = obj["objectId"]
    otype = obj["objectType"]

    event = controller.step(action="SliceObject", objectId=oid, forceAction=True)
    if event.metadata["lastActionSuccess"]:
        slice_results["success"].append(otype)
    else:
        slice_results["fail"].append((otype, event.metadata.get("errorMessage", "")[:60]))

print(f"  ✓ Can slice ({len(slice_results['success'])}): {', '.join(sorted(set(slice_results['success'])))}")
print(f"  ✗ Slice fails ({len(slice_results['fail'])}):")
for name, err in slice_results["fail"]:
    print(f"      {name}: {err}")

# ── 10. COFFEEMACHINE / SINK / FAUCET FILL TEST ──
print(f"\n{'─'*70}")
print("10. COFFEEMACHINE & SINK/FAUCET FILL TEST")
print(f"{'─'*70}")

controller.reset(scene=SCENE)
time.sleep(0.2)
objects = controller.last_event.metadata["objects"]

appliance_fill_results = []

# --- CoffeeMachine workflow ---
# Workflow: put mug in CoffeeMachine → toggle on → check if mug filled
coffee_machine = next((o for o in objects if o["objectType"] == "CoffeeMachine"), None)
if coffee_machine:
    # Test with Mug
    mug = next((o for o in objects if o["objectType"] == "Mug"), None)
    if mug:
        controller.reset(scene=SCENE)
        time.sleep(0.2)
        objects = controller.last_event.metadata["objects"]
        coffee_machine = next((o for o in objects if o["objectType"] == "CoffeeMachine"), None)
        mug = next((o for o in objects if o["objectType"] == "Mug"), None)

        # Pick up mug
        ev = controller.step(action="PickupObject", objectId=mug["objectId"], forceAction=True)
        print(f"  [CoffeeMachine] Pick up Mug: {ev.metadata['lastActionSuccess']}")

        # Put mug in CoffeeMachine
        ev = controller.step(action="PutObject", objectId=coffee_machine["objectId"], forceAction=True)
        print(f"  [CoffeeMachine] Put Mug in CoffeeMachine: {ev.metadata['lastActionSuccess']}")

        if ev.metadata["lastActionSuccess"]:
            # Toggle CoffeeMachine on
            ev = controller.step(action="ToggleObjectOn", objectId=coffee_machine["objectId"], forceAction=True)
            print(f"  [CoffeeMachine] Toggle ON: {ev.metadata['lastActionSuccess']}")

            if ev.metadata["lastActionSuccess"]:
                time.sleep(0.5)
                # Check mug state
                objects = controller.last_event.metadata["objects"]
                mug_after = next((o for o in objects if o["objectType"] == "Mug"), None)
                filled = mug_after.get("isFilledWithLiquid", False) if mug_after else False
                fill_liquid = mug_after.get("fillLiquid", None) if mug_after else None
                print(f"  [CoffeeMachine] Mug filled: {filled}, liquid: {fill_liquid}")
                appliance_fill_results.append(("CoffeeMachine+Mug", filled, fill_liquid))

                # Toggle off
                controller.step(action="ToggleObjectOff", objectId=coffee_machine["objectId"], forceAction=True)
            else:
                appliance_fill_results.append(("CoffeeMachine+Mug", False, "toggle failed"))
        else:
            err = ev.metadata.get("errorMessage", "")[:80]
            print(f"  [CoffeeMachine] Mug placement failed: {err}")
            appliance_fill_results.append(("CoffeeMachine+Mug", False, f"placement: {err}"))

    # Test with Cup
    controller.reset(scene=SCENE)
    time.sleep(0.2)
    objects = controller.last_event.metadata["objects"]
    coffee_machine = next((o for o in objects if o["objectType"] == "CoffeeMachine"), None)
    cup = next((o for o in objects if o["objectType"] == "Cup"), None)
    if cup:
        ev = controller.step(action="PickupObject", objectId=cup["objectId"], forceAction=True)
        ev = controller.step(action="PutObject", objectId=coffee_machine["objectId"], forceAction=True)
        print(f"  [CoffeeMachine] Put Cup in CoffeeMachine: {ev.metadata['lastActionSuccess']}")
        if ev.metadata["lastActionSuccess"]:
            ev = controller.step(action="ToggleObjectOn", objectId=coffee_machine["objectId"], forceAction=True)
            time.sleep(0.5)
            objects = controller.last_event.metadata["objects"]
            cup_after = next((o for o in objects if o["objectType"] == "Cup"), None)
            filled = cup_after.get("isFilledWithLiquid", False) if cup_after else False
            fill_liquid = cup_after.get("fillLiquid", None) if cup_after else None
            print(f"  [CoffeeMachine] Cup filled: {filled}, liquid: {fill_liquid}")
            appliance_fill_results.append(("CoffeeMachine+Cup", filled, fill_liquid))
            controller.step(action="ToggleObjectOff", objectId=coffee_machine["objectId"], forceAction=True)
        else:
            appliance_fill_results.append(("CoffeeMachine+Cup", False, "placement failed"))

    # Test with Bowl
    controller.reset(scene=SCENE)
    time.sleep(0.2)
    objects = controller.last_event.metadata["objects"]
    coffee_machine = next((o for o in objects if o["objectType"] == "CoffeeMachine"), None)
    bowl = next((o for o in objects if o["objectType"] == "Bowl"), None)
    if bowl:
        ev = controller.step(action="PickupObject", objectId=bowl["objectId"], forceAction=True)
        ev = controller.step(action="PutObject", objectId=coffee_machine["objectId"], forceAction=True)
        print(f"  [CoffeeMachine] Put Bowl in CoffeeMachine: {ev.metadata['lastActionSuccess']}")
        appliance_fill_results.append(("CoffeeMachine+Bowl", ev.metadata["lastActionSuccess"], "placement test"))
else:
    print("  [CoffeeMachine] Not in scene — skipping")

# --- Sink/Faucet fill workflow ---
# Workflow: hold object under faucet → toggle faucet on → check if object fills
print()
controller.reset(scene=SCENE)
time.sleep(0.2)
objects = controller.last_event.metadata["objects"]

faucet = next((o for o in objects if o["objectType"] == "Faucet"), None)
sink_basin = next((o for o in objects if o["objectType"] == "SinkBasin"), None)

if faucet and sink_basin:
    fill_test_items = ["Mug", "Cup", "Bowl", "Pot", "Bottle", "WateringCan", "Kettle"]
    for item_type in fill_test_items:
        controller.reset(scene=SCENE)
        time.sleep(0.1)
        objects = controller.last_event.metadata["objects"]
        faucet = next((o for o in objects if o["objectType"] == "Faucet"), None)
        sink_basin = next((o for o in objects if o["objectType"] == "SinkBasin"), None)
        item = next((o for o in objects if o["objectType"] == item_type), None)

        if not item:
            print(f"  [Sink] {item_type}: not in scene")
            appliance_fill_results.append((f"Sink+{item_type}", False, "not in scene"))
            continue

        # Pick up item
        ev = controller.step(action="PickupObject", objectId=item["objectId"], forceAction=True)
        if not ev.metadata["lastActionSuccess"]:
            print(f"  [Sink] {item_type}: can't pick up")
            appliance_fill_results.append((f"Sink+{item_type}", False, "can't pick up"))
            continue

        # Put item in SinkBasin
        ev = controller.step(action="PutObject", objectId=sink_basin["objectId"], forceAction=True)
        if not ev.metadata["lastActionSuccess"]:
            err = ev.metadata.get("errorMessage", "")[:60]
            print(f"  [Sink] {item_type} → SinkBasin: FAIL ({err})")
            appliance_fill_results.append((f"Sink+{item_type}", False, f"place fail: {err}"))
            controller.step(action="DropHandObject", forceAction=True)
            continue

        # Turn faucet on
        ev = controller.step(action="ToggleObjectOn", objectId=faucet["objectId"], forceAction=True)
        print(f"  [Sink] Faucet ON with {item_type} in basin: {ev.metadata['lastActionSuccess']}")

        if ev.metadata["lastActionSuccess"]:
            time.sleep(0.5)
            objects = controller.last_event.metadata["objects"]
            item_after = next((o for o in objects if o["objectType"] == item_type), None)
            filled = item_after.get("isFilledWithLiquid", False) if item_after else False
            fill_liquid = item_after.get("fillLiquid", None) if item_after else None
            dirty = item_after.get("isDirty", None) if item_after else None
            print(f"  [Sink] {item_type} filled: {filled}, liquid: {fill_liquid}, dirty: {dirty}")
            appliance_fill_results.append((f"Sink+{item_type}", filled, fill_liquid))

            # Toggle faucet off
            controller.step(action="ToggleObjectOff", objectId=faucet["objectId"], forceAction=True)
        else:
            appliance_fill_results.append((f"Sink+{item_type}", False, "faucet toggle failed"))
else:
    print("  [Sink] Faucet or SinkBasin not in scene — skipping")

print(f"\n  Summary of appliance fill tests:")
for combo, success, detail in appliance_fill_results:
    icon = "✓" if success else "✗"
    print(f"    {icon} {combo:30s}  filled={success}  detail={detail}")

# ── 11. BREAK TEST v2 — Alternative methods ──
print(f"\n{'─'*70}")
print("11. BREAK TEST v2 — BreakObject action + Window smashing")
print(f"{'─'*70}")

controller.reset(scene=SCENE)
time.sleep(0.2)
objects = controller.last_event.metadata["objects"]
breakable = [o for o in objects if o.get("breakable")]
break_v2_results = []

# Method 1: Direct BreakObject action (if it exists)
for obj in breakable[:5]:  # Test first 5 to save time
    controller.reset(scene=SCENE)
    time.sleep(0.1)
    oid = obj["objectId"]
    otype = obj["objectType"]

    # Try BreakObject directly
    event = controller.step(action="BreakObject", objectId=oid, forceAction=True)
    if event.metadata["lastActionSuccess"]:
        break_v2_results.append((otype, "BreakObject", True, ""))
        print(f"  ✓ {otype}: BreakObject works!")
    else:
        err = event.metadata.get("errorMessage", "")[:80]
        break_v2_results.append((otype, "BreakObject", False, err))
        print(f"  ✗ {otype}: BreakObject — {err}")

# Method 2: ThrowObject with higher force
for obj in breakable[:5]:
    controller.reset(scene=SCENE)
    time.sleep(0.1)
    oid = obj["objectId"]
    otype = obj["objectType"]

    controller.step(action="PickupObject", objectId=oid, forceAction=True)
    for force in [300, 500, 1000]:
        event = controller.step(action="ThrowObject", moveMagnitude=force, forceAction=True)
        time.sleep(0.3)
        objects_after = controller.last_event.metadata["objects"]
        broken = any(o.get("isBroken") and o["objectType"] == otype for o in objects_after)
        if broken:
            break_v2_results.append((otype, f"Throw(force={force})", True, ""))
            print(f"  ✓ {otype}: ThrowObject force={force} broke it!")
            break
    else:
        break_v2_results.append((otype, "Throw(all forces)", False, "didn't break"))
        print(f"  ✗ {otype}: ThrowObject up to force=1000 — didn't break")

# Method 3: Try Window specifically (teammate mentioned it can be smashed)
window = next((o for o in objects if o["objectType"] == "Window"), None)
if window:
    controller.reset(scene=SCENE)
    time.sleep(0.1)
    objects = controller.last_event.metadata["objects"]
    window = next((o for o in objects if o["objectType"] == "Window"), None)

    # Check if Window is breakable
    print(f"\n  Window found: {window['objectId']}")
    print(f"    breakable={window.get('breakable')}, isBroken={window.get('isBroken')}")

    # Try BreakObject
    ev = controller.step(action="BreakObject", objectId=window["objectId"], forceAction=True)
    print(f"    BreakObject: success={ev.metadata['lastActionSuccess']}, err={ev.metadata.get('errorMessage', '')[:80]}")
    if ev.metadata["lastActionSuccess"]:
        break_v2_results.append(("Window", "BreakObject", True, ""))
    else:
        # Try throwing something at it
        # Pick up something heavy and throw toward window
        for throw_item_type in ["Mug", "Pot", "Pan", "Bowl", "Plate"]:
            controller.reset(scene=SCENE)
            time.sleep(0.1)
            objects = controller.last_event.metadata["objects"]
            throw_item = next((o for o in objects if o["objectType"] == throw_item_type), None)
            window = next((o for o in objects if o["objectType"] == "Window"), None)
            if throw_item:
                controller.step(action="PickupObject", objectId=throw_item["objectId"], forceAction=True)
                # Throw toward window direction
                ev = controller.step(action="ThrowObject", moveMagnitude=500, forceAction=True)
                time.sleep(0.5)
                objects_after = controller.last_event.metadata["objects"]
                w_after = next((o for o in objects_after if o["objectType"] == "Window"), None)
                if w_after and w_after.get("isBroken"):
                    print(f"    Throwing {throw_item_type} broke Window!")
                    break_v2_results.append(("Window", f"Throw({throw_item_type})", True, ""))
                    break
        else:
            print(f"    Window could not be broken with any method")
            break_v2_results.append(("Window", "all methods", False, ""))
else:
    print("  No Window in scene")

# ── 12. BREADSLICED + TOASTER TEST ──
print(f"\n{'─'*70}")
print("12. BREADSLICED → TOASTER TEST — Slice bread, then toast it")
print(f"{'─'*70}")

controller.reset(scene=SCENE)
time.sleep(0.2)
objects = controller.last_event.metadata["objects"]

bread = next((o for o in objects if o["objectType"] == "Bread"), None)
toaster = next((o for o in objects if o["objectType"] == "Toaster"), None)

breadsliced_results = {}
if bread and toaster:
    # Step 1: Slice the bread
    ev = controller.step(action="SliceObject", objectId=bread["objectId"], forceAction=True)
    print(f"  Slice Bread: {ev.metadata['lastActionSuccess']}")

    if ev.metadata["lastActionSuccess"]:
        time.sleep(0.3)
        objects = controller.last_event.metadata["objects"]
        # Find BreadSliced objects
        bread_slices = [o for o in objects if o["objectType"] == "BreadSliced"]
        print(f"  Found {len(bread_slices)} BreadSliced objects")

        if bread_slices:
            slice_obj = bread_slices[0]
            # Pick up a slice
            ev = controller.step(action="PickupObject", objectId=slice_obj["objectId"], forceAction=True)
            print(f"  Pick up BreadSliced: {ev.metadata['lastActionSuccess']}")

            if ev.metadata["lastActionSuccess"]:
                # Put in toaster
                ev = controller.step(action="PutObject", objectId=toaster["objectId"], forceAction=True)
                print(f"  Put BreadSliced in Toaster: {ev.metadata['lastActionSuccess']}")
                if ev.metadata["lastActionSuccess"]:
                    # Toggle toaster on
                    ev = controller.step(action="ToggleObjectOn", objectId=toaster["objectId"], forceAction=True)
                    print(f"  Toggle Toaster ON: {ev.metadata['lastActionSuccess']}")
                    if ev.metadata["lastActionSuccess"]:
                        time.sleep(0.5)
                        objects = controller.last_event.metadata["objects"]
                        slice_after = next((o for o in objects if o["objectId"] == slice_obj["objectId"]), None)
                        cooked = slice_after.get("isCooked", False) if slice_after else None
                        print(f"  BreadSliced cooked: {cooked}")
                        breadsliced_results = {"slice": True, "place": True, "toast": True, "cooked": cooked}
                    else:
                        breadsliced_results = {"slice": True, "place": True, "toast": False}
                else:
                    err = ev.metadata.get("errorMessage", "")[:80]
                    print(f"  Placement failed: {err}")
                    breadsliced_results = {"slice": True, "place": False, "error": err}
    else:
        breadsliced_results = {"slice": False}
else:
    missing = []
    if not bread: missing.append("Bread")
    if not toaster: missing.append("Toaster")
    print(f"  Missing: {', '.join(missing)}")
    breadsliced_results = {"skip": True, "missing": missing}

print(f"  Result: {breadsliced_results}")

# ── 13. DIRTY/CLEAN TEST ──
print(f"\n{'─'*70}")
print("13. DIRTY/CLEAN TEST — Which objects can be dirtied and cleaned?")
print(f"{'─'*70}")

controller.reset(scene=SCENE)
time.sleep(0.2)
objects = controller.last_event.metadata["objects"]
dirtyable = [o for o in objects if o.get("dirtyable")]
dirty_results = []

for obj in dirtyable:
    oid = obj["objectId"]
    otype = obj["objectType"]

    ev = controller.step(action="DirtyObject", objectId=oid, forceAction=True)
    if ev.metadata["lastActionSuccess"]:
        # Try clean
        ev2 = controller.step(action="CleanObject", objectId=oid, forceAction=True)
        dirty_results.append((otype, True, ev2.metadata["lastActionSuccess"]))
    else:
        dirty_results.append((otype, False, False))

if dirty_results:
    for otype, can_dirty, can_clean in dirty_results:
        print(f"  {otype:20s}  dirty={'✓' if can_dirty else '✗'}  clean={'✓' if can_clean else '✗'}")
else:
    print("  No dirtyable objects in scene")

# ── SUMMARY ──
print(f"\n{'='*70}")
print("  DIAGNOSTIC SUMMARY")
print(f"{'='*70}")
print(f"  Scene: {SCENE}")
print(f"  Total objects: {len(by_type)} types, {sum(len(v) for v in by_type.values())} instances")
print(f"  Pickupable: {len(pickup_results['success'])} ok, {len(pickup_results['fail'])} fail")
print(f"  Toggleable: {len(toggle_results['on_success'])} ok, {len(toggle_results['on_fail'])} fail")
print(f"  Openable: {len(open_results['open_ok'])} ok, {len(open_results['open_fail'])} fail")
print(f"  Fillable: {len(fill_results['success'])} combos ok, {len(fill_results['fail'])} fail")
place_ok = sum(1 for _, _, s, _ in placement_results if s == "OK")
place_fail = sum(1 for _, _, s, _ in placement_results if s == "FAIL")
place_skip = sum(1 for _, _, s, _ in placement_results if s == "SKIP")
print(f"  Placements: {place_ok} ok, {place_fail} fail, {place_skip} skipped (not in scene)")
print(f"  Breakable: {len(break_results['success'])} ok, {len(break_results['fail'])} unclear")
print(f"  Sliceable: {len(slice_results['success'])} ok, {len(slice_results['fail'])} fail")
print(f"  Appliance fills: {sum(1 for _, s, _ in appliance_fill_results if s)} ok, {sum(1 for _, s, _ in appliance_fill_results if not s)} fail")
print(f"  Break v2: {sum(1 for _, _, s, _ in break_v2_results if s)} ok, {sum(1 for _, _, s, _ in break_v2_results if not s)} fail")
print(f"  BreadSliced→Toaster: {breadsliced_results}")
print(f"  Dirtyable: {sum(1 for _, d, _ in dirty_results if d)} dirty ok, {sum(1 for _, _, c in dirty_results if c)} clean ok")

# Save results to JSON
results = {
    "scene": SCENE,
    "object_types": {t: len(v) for t, v in by_type.items()},
    "pickup": {"ok": list(set(pickup_results["success"])), "fail": [n for n, _ in pickup_results["fail"]]},
    "toggle": {"on_ok": list(set(toggle_results["on_success"])), "on_fail": [n for n, _ in toggle_results["on_fail"]]},
    "open": {"ok": list(set(open_results["open_ok"])), "fail": [n for n, _ in open_results["open_fail"]]},
    "fill": {"ok": fill_results["success"], "fail": fill_results["fail"]},
    "placements": [{"item": i, "recep": r, "status": s, "detail": d} for i, r, s, d in placement_results],
    "break": {"ok": list(set(break_results["success"])), "unclear": list(set(break_results["fail"]))},
    "slice": {"ok": list(set(slice_results["success"])), "fail": [n for n, _ in slice_results["fail"]]},
    "appliance_fill": [{"combo": c, "filled": s, "detail": str(d)} for c, s, d in appliance_fill_results],
    "break_v2": [{"object": o, "method": m, "success": s, "detail": d} for o, m, s, d in break_v2_results],
    "breadsliced_toaster": breadsliced_results,
    "dirty_clean": [{"type": t, "can_dirty": d, "can_clean": c} for t, d, c in dirty_results],
}

outfile = f"diagnostic_{SCENE}.json"
with open(outfile, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Results saved to: {outfile}")

# Don't call controller.stop() — we're connected to an external Unity server
print("\nDone.")
