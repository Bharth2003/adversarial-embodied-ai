# Main & CLI

> **Files:** `src/ai2thor_lab/main.py`, `src/ai2thor_lab/cli.py`

These are the two **entry points** into the system.

---

## `main.py` — Task Runner

The main entry point. Parses CLI arguments and dispatches to the appropriate pipeline:

```
python -m ai2thor_lab.main [--scene FloorPlan1] [--task "..."] [--plan]
```

| Arguments | Behavior |
|---|---|
| No `--task` | Launches **interactive mode** (human types commands) |
| `--task "fill pot with water"` | Runs the **reactive LLM loop** (`LLMWrapper.run()`) |
| `--task "fill pot with water" --plan` | Runs the **deterministic pipeline** (`GoalParser` → `DeterministicPlanner` → `PlanExecutor`) |

### Deterministic Pipeline Flow (with `--plan`)

1. `GoalParser.parse(task, scene_metadata)` → structured goals (1 LLM call)
2. `DeterministicPlanner(scene_metadata).plan(goals)` → action plan (instant, no LLM)
3. `DeterministicPlanner.print_plan(plan)` → prints the plan
4. `PlanExecutor(agent).execute_plan(plan)` → executes against the simulator

Each run generates a unique **Langfuse session ID** for observability tracing.

---

## `cli.py` — Interactive Mode

A terminal REPL for manually controlling the agent with text commands:

```
> forward 0.5        # move forward
> open fridge        # open the fridge
> pick up egg        # pick up the egg
> look               # list visible objects
> where              # print agent position
> metadata           # dump full scene JSON
> help               # show all available commands
> quit               # exit
```

### Special Commands

| Command | Effect |
|---|---|
| `look` / `list` | Lists all visible objects with properties, distances, and states |
| `where` | Prints agent position, rotation, and camera horizon |
| `metadata` | Dumps the full AI2-THOR event metadata as JSON |
| `help` | Prints the command reference |
| `quit` / `exit` / `q` | Exits the program |

Everything else is passed to `agent.execute_command()` for parsing and execution.
