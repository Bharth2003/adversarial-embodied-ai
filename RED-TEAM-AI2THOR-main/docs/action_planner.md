# Action Planner

> **File:** `src/ai2thor_lab/action_planner.py`

The `ActionPlanner` is an **alternative planning approach** where the LLM directly generates a sequence of primitive AI2-THOR actions (instead of extracting abstract goals first like `GoalParser` + `DeterministicPlanner` does).

---

## Motivation

The goal-based pipeline (`GoalParser` → `DeterministicPlanner`) has an abstraction layer: the LLM extracts high-level goals, then deterministic code expands them. This works well for common tasks but can fail on unusual ones (e.g. *"dirty the mug"*, *"throw apple in garbage"*) where the goal types don't map cleanly.

The `ActionPlanner` removes this abstraction bottleneck — the LLM sees the full action space with preconditions and generates executable steps directly.

## How It Works

1. **Prompt construction** — builds a detailed system prompt containing:
   - All 16 available actions with their exact preconditions (e.g. *"slice: object must NOT be held, must be on a surface, you must hold a Knife"*)
   - Every object in the scene with its properties and current state
   - 10 planning rules covering common pitfalls

2. **Single LLM call** — uses OpenAI function calling with a `set_plan` tool that returns a structured list of steps.

3. **Format conversion** — normalizes the LLM output into the `PlanExecutor` format:
   - Maps LLM action names to executor tool names (`"put"` → `"place_on"`, `"fill"` → `"fill_with_liquid"`)
   - Ensures `object_id` / `receptacle_id` are in the right fields
   - Appends a `finish` step

## Comparison with Goal-Based Pipeline

| | GoalParser + DeterministicPlanner | ActionPlanner |
|---|---|---|
| LLM responsibility | Extract end-state goals only | Generate full action sequence |
| Planning logic | Deterministic code | LLM reasoning |
| Reliability | Higher (guaranteed-correct expansion) | Lower (LLM may produce invalid sequences) |
| Flexibility | Limited to known goal types | Can handle any task the LLM understands |
| LLM calls | 1 (goal parsing) | 1 (plan generation) |

## Output Format

Same as what `PlanExecutor` expects:
```json
{
  "steps": [
    {"step": 1, "tool": "navigate_to", "arguments": {"object_id": "Mug"}, "reason": "Go to mug"},
    {"step": 2, "tool": "dirty", "arguments": {"object_id": "Mug"}, "reason": "Dirty the mug"},
    {"step": 3, "tool": "finish", "arguments": {}, "reason": "Task complete"}
  ]
}
```
