# Plan Executor

> **File:** `src/ai2thor_lab/plan_executor.py`

The `PlanExecutor` takes a plan (a JSON dict of ordered steps) and **executes each step against the live AI2-THOR environment**, handling ID resolution, validation, error recovery, and post-action verification.

---

## Pipeline

```
Plan JSON ──→ PlanExecutor.execute_plan()
                 │
                 ├── for each step:
                 │     1. Resolve object IDs  (short name → full AI2-THOR ID)
                 │     2. Pre-action validation (visible? close enough?)
                 │     3. Execute via controller
                 │     4. Post-action verification (did state actually change?)
                 │
                 └── Return execution report
```

## Object ID Resolution

The planner works with **short names** like `"Pot"` or `"Fridge"`, but AI2-THOR needs full IDs like `"Pot|-01.22|+00.90|-02.36"`. The `_resolve_object_id` method tries six increasingly fuzzy strategies:

1. **Exact match** — the raw ID is already a full object ID
2. **objectType match** — `"Pot"` → first Pot in the scene
3. **Case-insensitive type** — `"pot"` → `"Pot"`
4. **Substring in ID** — partial match against the full ID string
5. **Case-insensitive substring** — same but case-folded
6. **Name field** — match against the object's `name` property

## Pre-Action Validation

Before executing most tools, the executor checks:
- **Exists** — object is in the scene
- **Visible** — object is currently visible to the agent
- **In range** — object is within 1.5m interaction distance
- **State preconditions** — e.g. skips `open` if already open, skips `pickup` if already holding it

## Auto-Recovery

If the agent navigated to an object but can't see it, `_auto_recover_rotation` does a full 360° rotation scan (at multiple look-down angles) to try to find the target before giving up.

## Supported Tools

The executor handles every tool from `tools.py`: `navigate_to`, `pickup`, `drop`, `place_on`, `open`, `close`, `toggle_on/off`, `fill_with_liquid`, `empty_liquid`, `cook`, `slice`, `clean`, `dirty`, `break_object`, `use_up`, `finish`, plus movement/rotation/look commands.

## Execution Report

`execute_plan()` returns:
```python
{"success": True, "report": [...]}           # all steps passed
{"success": False, "failed_step": 3,          # halted at step 3
 "reason": "Object not visible", "report": [...]}
```

Execution **halts on the first failure** — it does not attempt to continue past a failed step.
