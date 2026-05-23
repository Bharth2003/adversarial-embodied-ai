# Planner (Legacy)

> **File:** `src/ai2thor_lab/planner.py`

The `Planner` class is an **earlier iteration** of the planning system. Like `ActionPlanner`, it asks the LLM to generate action steps directly, but with a simpler prompt and plain JSON output (no function calling).

---

## How It Differs

| | `Planner` (this file) | `ActionPlanner` | `GoalParser` + `DeterministicPlanner` |
|---|---|---|---|
| LLM output format | Raw JSON in message content | Function-calling tool (`set_plan`) | Function-calling tool (`set_goals`) + deterministic code |
| Action preconditions in prompt | Minimal (5 rules) | Detailed (per-action preconditions) | N/A (handled by code) |
| Response parsing | Strip markdown, `json.loads` | Extract from tool call | Extract from tool call |

## Key Method

`generate_plan(task, scene_metadata)` → `{"plan": {...}, "success": True}`

1. Builds a planning prompt with the task, available objects, tool list, and example JSON output.
2. Calls the LLM with **no tools** — expects the model to return raw JSON in the message content.
3. Strips any markdown code-block fencing and parses the JSON.
4. Returns the plan wrapped in `{"plan": ..., "success": True}`.

## Notes

- The prompt includes a full worked example (fill pot → place on stove → toggle knob) to guide the LLM.
- Uses `self.tool_schemas` and `self.tool_names` from `tools.py` for the prompt, but doesn't pass them as callable tools to the LLM.
- `print_plan()` pretty-prints the plan with step numbers and reasons.
