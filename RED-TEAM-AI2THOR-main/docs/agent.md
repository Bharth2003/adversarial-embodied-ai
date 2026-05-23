# Agent

> **File:** `src/ai2thor_lab/agent.py`

The `Agent` class is the central interface between the codebase and the **AI2-THOR simulator**. It wraps the AI2-THOR `Controller`, manages held-object state, and provides helpers that the rest of the system uses to query the scene and execute low-level commands.

---

## Responsibilities

| Method | What it does |
|---|---|
| `__init__(scene)` | Launches the AI2-THOR controller with the given floor plan and default camera/grid settings. Creates the `CommandParser`. |
| `get_visible_objects()` | Returns every **currently visible** object with a rich property dict (pickupable, openable, toggleable, all state booleans, etc.). |
| `get_all_objects()` | Returns **every object in the scene** regardless of visibility — used by the planner so it can reason about objects it can't currently see. |
| `get_scene_metadata()` | Bundles the agent's position/rotation/horizon and the full object list into one dict for the planners. |
| `execute_command(cmd)` | Parses a natural-language command string via `CommandParser`, sends the resulting action to the controller, tracks held-object state, and displays the updated frame. |
| `display_frame()` | Renders the current camera view in an OpenCV window with an HUD overlay (position, rotation, held object). |
| `get_context_description()` | Builds a plain-text description of the agent's state and visible objects — used as observation input for the LLM loop. |
| `get_agent_position()` | Returns a dict with `x, y, z, rotation, horizon`. |
| `get_action_result()` | Returns a structured dict with `success`, `error`, `position`, and `held_object` after the last action. |

## Held-Object Tracking

The agent manually tracks `self.held_object` because the simulator doesn't expose a convenient "what am I holding?" field. It updates on:
- **PickupObject** → sets `held_object` to the picked-up ID
- **DropHandObject / PutObject** → clears `held_object`
- **BreakObject / SliceObject** → clears if the held object was the one broken/sliced (it gets destroyed)

## Key Design Note

The `Agent` deliberately exposes **two object views**:
- `get_visible_objects()` — for the reactive LLM loop, which should only reason about what it can see.
- `get_all_objects()` — for the deterministic planner, which needs the full scene graph to plan ahead.
