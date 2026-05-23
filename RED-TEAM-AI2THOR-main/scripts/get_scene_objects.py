"""
Get all unique object types from the running AI2-THOR scene and re-filter datasets.
Run this while the server is running: uv run python get_scene_objects.py
"""
import requests
import json
import os

SERVER = "http://localhost:8000"

# Step 1: Get all objects from the scene
print("Fetching all objects from scene...")
resp = requests.get(f"{SERVER}/all_objects")
data = resp.json()

scene = data["scene"]
object_types = data["object_types"]

print(f"Scene: {scene}")
print(f"Total unique object types: {data['total']}")
print("\nObjects in scene:")
for t in object_types:
    print(f"  {t}")

# Save object list
objects_file = "../data/floorplan_messy_objects.txt"
with open(objects_file, "w") as f:
    for t in object_types:
        f.write(t + "\n")
print(f"\nSaved to {objects_file}")

# Step 2: Filter datasets using the same logic as step3_filter_dataset.py
OBJECT_NAME_FIXES = {
    "AlarmCLock": "AlarmClock",
    "CellPhones": "CellPhone",
    "Cellphone": "CellPhone",
    "Objects: Bowl": "Bowl",
    "Objects: Statue": "Statue",
    "Coffee": "CoffeeMachine",
    "Wine": "WineBottle",
    "Water": None,
    "Appliances": None,
}

# Only skip objects that are truly universal / not interactable targets.
# Do NOT skip furniture like DiningTable, SideTable, Desk etc. — they may not exist in all scenes.
STRUCTURAL_OBJECTS_LOWER = {s.lower() for s in {
    "Floor", "Wall", "Window", "LightSwitch",
}}

messy_lookup = {t.lower(): t for t in object_types}

def fix_name(name):
    name = name.strip()
    return OBJECT_NAME_FIXES.get(name, name)

def extract_objects_safe(entry):
    objects = set()
    if entry.get("final_state"):
        for state in entry["final_state"]:
            if "objectType" in state:
                objects.add(state["objectType"])
            for recep in (state.get("parentReceptacles") or []):
                objects.add(recep)
    return objects

def extract_objects_unsafe(entry):
    objects = set()
    if entry.get("objects"):
        for obj in entry["objects"]:
            objects.add(obj)
    if entry.get("final_state"):
        for state in entry["final_state"]:
            if "objectType" in state:
                objects.add(state["objectType"])
            for recep in (state.get("parentReceptacles") or []):
                objects.add(recep)
    return objects

def filter_dataset(input_path, output_path, extract_fn):
    kept, dropped = 0, 0
    drop_reasons = {}

    if not os.path.exists(input_path):
        print(f"  SKIP: {input_path} not found")
        return 0, 0, {}

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            entry = json.loads(line)
            raw_objects = extract_fn(entry)
            fixed_objects = set()
            for obj in raw_objects:
                fixed = fix_name(obj)
                if fixed is None:
                    continue
                fixed_objects.add(fixed)

            missing = set()
            for obj in fixed_objects:
                if obj.lower() in STRUCTURAL_OBJECTS_LOWER:
                    continue
                if obj.lower() not in messy_lookup:
                    missing.add(obj)

            if missing:
                dropped += 1
                for m in missing:
                    drop_reasons[m] = drop_reasons.get(m, 0) + 1
            else:
                entry["scene_name"] = scene
                fout.write(json.dumps(entry) + "\n")
                kept += 1

    return kept, dropped, drop_reasons

# Check for source datasets in multiple locations
dataset_dir = "../dataset"
adversarial_dir = os.path.join("..", "..", "..", "adversarial-embodied-safety-main_vlm", "adversarial-embodied-safety-main", "dataset")

# Find the full 1009 datasets
safe_input = None
unsafe_input = None

for d in [dataset_dir, adversarial_dir]:
    s = os.path.join(d, "safe_detailed_1009.jsonl")
    u = os.path.join(d, "unsafe_detailed_1009.jsonl")
    if os.path.exists(s) and safe_input is None:
        safe_input = s
    if os.path.exists(u) and unsafe_input is None:
        unsafe_input = u

if not safe_input or not unsafe_input:
    print("\nERROR: Could not find safe_detailed_1009.jsonl and unsafe_detailed_1009.jsonl")
    print("Checked:", dataset_dir, "and", adversarial_dir)
    exit(1)

print(f"\nSource datasets:")
print(f"  Safe: {safe_input}")
print(f"  Unsafe: {unsafe_input}")

safe_output = os.path.join(dataset_dir, "safe_detailed_messy.jsonl")
unsafe_output = os.path.join(dataset_dir, "unsafe_detailed_messy.jsonl")

print(f"\nFiltering safe dataset...")
safe_kept, safe_dropped, safe_reasons = filter_dataset(safe_input, safe_output, extract_objects_safe)
print(f"  Kept: {safe_kept} / {safe_kept + safe_dropped}")
if safe_reasons:
    print(f"  Top missing objects:")
    for obj, count in sorted(safe_reasons.items(), key=lambda x: -x[1])[:10]:
        print(f"    {obj}: {count} entries")

print(f"\nFiltering unsafe dataset...")
unsafe_kept, unsafe_dropped, unsafe_reasons = filter_dataset(unsafe_input, unsafe_output, extract_objects_unsafe)
print(f"  Kept: {unsafe_kept} / {unsafe_kept + unsafe_dropped}")
if unsafe_reasons:
    print(f"  Top missing objects:")
    for obj, count in sorted(unsafe_reasons.items(), key=lambda x: -x[1])[:10]:
        print(f"    {obj}: {count} entries")

print(f"\n{'='*60}")
print(f"DONE")
print(f"{'='*60}")
print(f"Safe:   {safe_kept} entries -> {safe_output}")
print(f"Unsafe: {unsafe_kept} entries -> {unsafe_output}")
print(f"\nRestart the server to use the filtered datasets.")
