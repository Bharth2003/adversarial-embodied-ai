# Object Types

> **File:** `src/ai2thor_lab/object_types.py`

A **reference list** of all object types available in AI2-THOR's iTHOR environment.

---

## Contents

- **`ITHOR_OBJECT_TYPES`** — a list of 130+ object type strings, from `"AlarmClock"` to `"WineBottle"`. Items marked with `*` (e.g. `"AppleSliced*"`, `"EggCracked*"`) represent state variations that appear after slicing/cracking — they shouldn't be used as planning targets since the planner handles state transitions on the base objects automatically.

- **`BASE_OBJECT_TYPES`** — a convenience set with the `*` suffixes stripped, useful for validation (checking whether a name refers to a valid base object).

## Purpose

This file serves as a static reference. It's not imported by the core pipeline but is useful for:
- Validating object names in prompts
- Building object catalogues for documentation
- Quick lookup of what's available in the simulator

Source: [AI2-THOR Object Types Documentation](https://ai2thor.allenai.org/ithor/documentation/objects/object-types)
