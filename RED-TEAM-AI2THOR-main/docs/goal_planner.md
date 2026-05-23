# Goal Planner

The goal planner converts **natural-language household tasks** (e.g. *"boil water in a pot"*) into a **deterministic sequence of robot actions** that can be executed in AI2-THOR.

It is split into two distinct stages so that the LLM is only used once (for understanding), while all planning logic is pure, instant, and guaranteed-correct code.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                         User Task                                │
│                  "open fridge, take egg,                          │
│                   place on countertop"                            │
└───────────────────────────┬──────────────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────┐
              │       GoalParser        │   ← single LLM call
              │  (natural language →    │
              │   structured goals)     │
              └────────────┬────────────┘
                           │
            List[Goal]:    │  e.g.  Goal(type="on", object="Egg",
                           │              receptacle="CounterTop")
                           ▼
              ┌─────────────────────────┐
              │  DeterministicPlanner   │   ← pure code, no LLM
              │    (goals → action      │
              │     sequence)           │
              └────────────┬────────────┘
                           │
                           ▼
              Ordered list of PlanSteps
              (navigate, pickup, open, place, …)
```

---

## Stage 1 — `GoalParser`

**Purpose:** Translate a free-text task into a list of `Goal` objects.

**How it works:**

1. Builds a **scene-aware prompt** listing every object in the current AI2-THOR scene along with its properties (pickupable, receptacle) and location.
2. Sends a single request to the LLM with a [function-calling tool](https://platform.openai.com/docs/guides/function-calling) called `set_goals`. This forces the LLM to return structured JSON rather than free text.
3. Extracts the goals from the tool-call response (with a fallback to raw JSON / markdown-fenced JSON parsing).
4. Retries once on failure for robustness.

### Goal Types

| Goal type     | Meaning                           | Extra fields         |
|---------------|-----------------------------------|----------------------|
| `holding`     | Agent is holding the object       | —                    |
| `on`          | Object is on a receptacle         | `receptacle`         |
| `filled`      | Object is filled with liquid      | `liquid` (water/coffee/wine) |
| `toggled_on`  | Appliance is turned on            | —                    |
| `toggled_off` | Appliance is turned off           | —                    |
| `open`        | Object is open                    | —                    |
| `closed`      | Object is closed                  | —                    |
| `cooked`      | Object has been cooked            | —                    |
| `sliced`      | Object has been sliced/cracked    | —                    |
| `cleaned`     | Object has been cleaned           | —                    |
| `dirty`       | Object has been dirtied           | —                    |
| `emptied`     | Object has been emptied           | —                    |
| `used_up`     | Object has been used up           | —                    |
| `broken`      | Object has been broken            | —                    |

### Key Design Decisions

- The LLM is instructed to extract **only end-state goals**, not intermediate steps. For example, *"get egg from fridge"* becomes `holding(Egg)` — the planner figures out it needs to open the fridge first.
- Object names must match the scene's vocabulary exactly (e.g. `Pot`, `StoveBurner`). The prompt lists all available objects to prevent hallucination.

---

## Stage 2 — `DeterministicPlanner`

**Purpose:** Convert `Goal` objects into an ordered list of executable `PlanStep` actions, handling all preconditions automatically.

**Key idea:** This is a [PDDL](https://en.wikipedia.org/wiki/Planning_Domain_Definition_Language)-inspired planner written entirely in code. It models action preconditions & effects so it can insert prerequisite steps automatically.

### Initialisation

On construction, the planner indexes the scene:

- **`objects`** — name → object properties
- **`objects_by_id`** — full AI2-THOR ID → properties
- **`containment`** — which objects are inside openable containers (e.g. `Egg` → `Fridge`)

It also tracks simulated agent state during planning:

- `holding` — what the agent currently holds (or `None`)
- `agent_near` — which objects the agent is near
- `opened_containers` — which containers have been opened
- `sliced` — which objects have been sliced

### Precondition Helpers

These **`_ensure_*`** methods model action preconditions. They add prerequisite steps only when needed:

| Helper                     | What it does                                                                                      |
|----------------------------|---------------------------------------------------------------------------------------------------|
| `_ensure_near(obj)`        | Adds a `navigate_to` step if not already near the object                                          |
| `_ensure_holding(obj)`     | Puts down current object if hands are full, opens containers if needed, navigates, and picks up   |
| `_ensure_open(obj)`        | Opens an openable object (fridge, cabinet) if not already open                                    |
| `_ensure_closed(obj)`      | Closes an object if it was opened                                                                 |
| `_ensure_container_open()` | Opens the container an object is stored inside (e.g. opens the fridge to reach the egg)           |

### Goal Decomposition (`_plan_goal`)

Each goal type maps to a specific recipe of actions. Some examples:

- **`on(Egg, CounterTop)`** → ensure holding Egg → navigate to CounterTop → place
- **`cooked(Egg)`** → if egg is not cookable, crack it first (slice) → pick up cracked version → cook
- **`filled(Pot, water)`** → ensure holding Pot → fill with liquid
- **`sliced(Apple)`** → if holding it, put it down first → navigate to it → slice
- **`broken(Vase)`** → if holding it, put it down first → navigate → break

### Plan Output

The `plan()` method iterates over all goals, concatenates the steps, appends a final `finish` step, and returns a dict:

```json
{
  "steps": [
    {"step": 1, "tool": "navigate_to", "arguments": {"object_id": "Fridge"}, "reason": "Go to Fridge"},
    {"step": 2, "tool": "open", "arguments": {"object_id": "Fridge"}, "reason": "Open Fridge to access Egg"},
    {"step": 3, "tool": "navigate_to", "arguments": {"object_id": "Egg"}, "reason": "Go to Egg"},
    {"step": 4, "tool": "pickup", "arguments": {"object_id": "Egg"}, "reason": "Pick up Egg"},
    {"step": 5, "tool": "navigate_to", "arguments": {"object_id": "CounterTop"}, "reason": "Go to CounterTop"},
    {"step": 6, "tool": "place_on", "arguments": {"receptacle_id": "CounterTop"}, "reason": "Place Egg on CounterTop"},
    {"step": 7, "tool": "finish", "arguments": {}, "reason": "Task complete"}
  ]
}
```

---

## How It Fits Into the System

```
main.py / cli.py
      │
      ▼
   agent.py  ──→  GoalParser  ──→  DeterministicPlanner
                      │                     │
                      │ (uses)              │ (output)
                      ▼                     ▼
                 llm_wrapper.py       plan_executor.py  ──→  tools.py / navigator.py
```

1. **`agent.py`** receives a task from the user.
2. **`GoalParser`** makes one LLM call to understand the task.
3. **`DeterministicPlanner`** instantly produces the action plan.
4. **`plan_executor.py`** executes each step against the AI2-THOR environment using the action tools in `tools.py` and navigation in `navigator.py`.
