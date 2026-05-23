# LLM Wrapper

> **File:** `src/ai2thor_lab/llm_wrapper.py`

The `LLMWrapper` implements a **reactive agent loop** — it repeatedly queries an LLM for the next action, executes it, observes the result, and feeds it back until the task is done (or the iteration limit is hit).

---

## Architecture

```
       ┌─────────────────────────────────┐
       │          LLMWrapper.run()       │
       │                                 │
       │   ┌──→ Call LLM (next action) ──┐
       │   │                             │
       │   │   Execute tool              │
       │   │                             │
       │   │   Get observation           │
       │   │                             │
       │   └── Append to history ────────┘
       │                                 │
       │   Loop until "finish" or max    │
       └─────────────────────────────────┘
```

## How the Loop Works

1. **System prompt** tells the LLM it's a robot agent with a specific goal, lists available tools, and gives usage rules.
2. Each iteration, the LLM returns a tool call (via OpenAI function calling) or a JSON object with `tool` and `arguments`.
3. The wrapper executes the tool against AI2-THOR (using direct controller calls + Navigator for movement).
4. The observation (visible objects, position, action result) is appended as a "user" message.
5. Repeat.

## Key Differences from Plan-Based Pipeline

| | LLMWrapper (reactive) | GoalPlanner + PlanExecutor (plan-based) |
|---|---|---|
| LLM calls | Many (one per action) | One (goal parsing only) |
| Adaptability | Adapts each step to observations | Fixed plan, halts on failure |
| Speed | Slow (LLM latency per step) | Fast (planning is instant) |
| Use flag | Default (`--task "..."`) | `--task "..." --plan` |

## Safety Features

- **Max iterations** — hard cap at 50 iterations to prevent infinite loops.
- **Consecutive failure tracking** — if 5 tool calls fail in a row, the loop aborts.

## `create_openai_client()`

Factory function at the bottom of the file that creates a reusable LLM client function. Features:
- Reads connection details from `.env` (Ollama URL, model, API key)
- Wraps calls with **Langfuse** observability (`@observe` decorator) for tracing
- Uses `propagate_attributes` to tag all calls with a session ID
- Returns a callable `(messages, tools) → ChatCompletion`
