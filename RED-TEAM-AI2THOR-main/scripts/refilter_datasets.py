"""Re-filter datasets using the saved object list. No server needed."""
import json
import os

# Load scene objects from saved file
with open("../data/floorplan_messy_objects.txt") as f:
    object_types = [line.strip() for line in f if line.strip()]

messy_lookup = {t.lower(): t for t in object_types}
print(f"Scene objects: {len(object_types)}")

OBJECT_NAME_FIXES = {
    "AlarmCLock": "AlarmClock", "CellPhones": "CellPhone", "Cellphone": "CellPhone",
    "Objects: Bowl": "Bowl", "Objects: Statue": "Statue",
    "Coffee": "CoffeeMachine", "Wine": "WineBottle", "Water": None, "Appliances": None,
}

# Only truly universal non-interactable objects
STRUCTURAL_LOWER = {"floor", "wall", "window", "lightswitch"}

def fix_name(name):
    return OBJECT_NAME_FIXES.get(name.strip(), name.strip())

def extract_from_steps(entry):
    """Extract object names from the step field (e.g. 'find DeskLamp' -> 'DeskLamp')."""
    objs = set()
    for step in (entry.get("step") or []):
        parts = step.strip().split()
        if len(parts) >= 2:
            # Actions like 'find X', 'pick X', 'open X', 'put X', 'turn_on X', etc.
            obj_name = parts[-1]
            # Skip liquid names and generic action words
            if obj_name.lower() not in {"water", "coffee", "wine", "on", "off"}:
                objs.add(obj_name)
    return objs

def extract_safe(entry):
    objs = set()
    for s in (entry.get("final_state") or []):
        if "objectType" in s: objs.add(s["objectType"])
        for r in (s.get("parentReceptacles") or []): objs.add(r)
    objs |= extract_from_steps(entry)
    return objs

def extract_unsafe(entry):
    objs = set()
    for o in (entry.get("objects") or []): objs.add(o)
    for s in (entry.get("final_state") or []):
        if "objectType" in s: objs.add(s["objectType"])
        for r in (s.get("parentReceptacles") or []): objs.add(r)
    objs |= extract_from_steps(entry)
    return objs

def filter_ds(inp, out, extract_fn):
    kept, dropped, reasons = 0, 0, {}
    if not os.path.exists(inp):
        print(f"  SKIP: {inp} not found"); return 0, 0, {}
    with open(inp) as fin, open(out, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            raw = extract_fn(entry)
            fixed = {fix_name(o) for o in raw if fix_name(o) is not None}
            missing = {o for o in fixed if o.lower() not in STRUCTURAL_LOWER and o.lower() not in messy_lookup}
            if missing:
                dropped += 1
                for m in missing: reasons[m] = reasons.get(m, 0) + 1
            else:
                entry["scene_name"] = "FloorPlan_Messy"
                fout.write(json.dumps(entry) + "\n"); kept += 1
    return kept, dropped, reasons

# Find source datasets
adv = os.path.join("..", "..", "..", "adversarial-embodied-safety-main_vlm", "adversarial-embodied-safety-main", "dataset")
for d in ["../dataset", adv]:
    s = os.path.join(d, "safe_detailed_1009.jsonl")
    u = os.path.join(d, "unsafe_detailed_1009.jsonl")
    if os.path.exists(s) and os.path.exists(u):
        safe_in, unsafe_in = s, u; break
else:
    print("ERROR: source datasets not found"); exit(1)

print(f"Source: {safe_in}, {unsafe_in}\n")

sk, sd, sr = filter_ds(safe_in, "../dataset/safe_detailed_messy.jsonl", extract_safe)
print(f"Safe:   {sk} kept / {sk+sd} total")
if sr: print(f"  Missing: {dict(sorted(sr.items(), key=lambda x:-x[1])[:10])}")

uk, ud, ur = filter_ds(unsafe_in, "../dataset/unsafe_detailed_messy.jsonl", extract_unsafe)
print(f"Unsafe: {uk} kept / {uk+ud} total")
if ur: print(f"  Missing: {dict(sorted(ur.items(), key=lambda x:-x[1])[:10])}")

print(f"\nDone. Restart server to use new datasets.")
