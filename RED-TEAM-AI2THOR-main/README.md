# ai2thor-lab

An adversarial AI safety benchmarking framework for embodied agents in [AI2-THOR](https://ai2thor.allenai.org/). Tests whether household robots can be tricked into performing dangerous actions through social engineering, deception, and prompt injection.

## Overview

This project implements a **Red Team vs Blue Team** framework where:

- **Red Agent (Attacker):** Uses LLM-powered social engineering, prompt injection, and deceptive framing to trick the robot into unsafe actions (e.g., putting metal in the microwave, breaking objects, hiding valuables).
- **Blue Agent (Defender):** A household robot with safety constitutions, a deterministic planner, and optional VLM (Vision Language Model) safety checks that must resist manipulation while completing legitimate tasks.
- **Judge:** Evaluates outcomes using ground-truth scene state (BlueTeamJudge) with LLM cross-checking.

The benchmark measures Attack Success Rate (ASR) — how often the red agent successfully jailbreaks the blue agent across diverse attack categories.

## Architecture

```
Red Agent (LLM)          Blue Agent (Planner + Safety)         Judge
     │                            │                              │
     ├─ Social engineering ──────>│                              │
     │                            ├─ Goal Parser (LLM)           │
     │                            ├─ Deterministic Planner       │
     │                            ├─ Safety Reviewer (Constitution)│
     │                            ├─ [Optional] VLM Safety Check │
     │                            ├─ Plan Executor (AI2-THOR)    │
     │                            │                              │
     │                            ├─ Scene state ───────────────>│
     │                            │                   BlueTeamJudge
     │                            │                   (ground truth)
```

## Requirements

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) (recommended package manager)
- AI2-THOR Unity build (see [Scene Setup](#scene-setup))
- [Ollama](https://ollama.ai/) running locally for LLM inference

## Installation

```bash
git clone <this-repo>
cd ai2thor-lab-main
uv sync
```

## Scene Setup

This project uses custom AI2-THOR scenes. The Unity builds are too large for GitHub, so you need to set them up separately.

### Option 1: Use default AI2-THOR scenes

The framework works with any AI2-THOR kitchen scene (FloorPlan1 through FloorPlan30). Default scenes are downloaded automatically by AI2-THOR.

### Option 2: Custom scenes (FloorPlan_Messy)

Our primary benchmark scene (`FloorPlan_Messy`) is a heavily cluttered kitchen with 228+ objects including stacked items, hidden objects in cabinets, and clustered duplicates. To use it:

1. Clone the AI2-THOR Unity source: https://github.com/allenai/ai2thor.git
2. Open the project in Unity (2020.3.25f1)
3. Load or create your custom scene
4. Build and run the Unity executable
5. The server connects to the Unity instance on `127.0.0.1:8200`

## Quick Start

### 1. Start the Unity server

Launch your AI2-THOR Unity build first. The server connects to it externally (`start_unity=False`).

### 2. Start the benchmark server

```bash
uv run python -m ai2thor_lab.server
```

This launches a FastAPI server with a web UI at `http://localhost:8000`.

### 3. Open the frontend

Open `frontend.html` in your browser. This provides the attack control panel with:

- Attack configuration (cases, turns, models, constitutions)
- Red agent mode selection (simple, Concordia, VLM, adaptive)
- Blue safety VLM toggle
- VLM jailbreak strategy selector
- Dual-robot mode toggle
- Real-time attack log with SSE events
- Leaderboard and statistics

### 4. Run attacks

Click "Start Attack" in the frontend. The system will:

1. Load the appropriate dataset based on the active scene
2. Cycle through attack cases
3. For each case: Red agent sends manipulation → Blue agent plans and executes (or refuses) → Judge evaluates scene state
4. Display results including ASR, per-category breakdowns, and leaderboard

## Configuration

### Environment Variables

```env
# LLM models (Ollama)
OLLAMA_MODEL="paulprt2:latest"
OLLAMA_URL="http://ubuntu:11434/v1"
OLLAMA_API_KEY="ollama"

# Optional: separate safety model
SAFETY_LLM_MODEL="paulprt2:latest"
SAFETY_LLM_URL="http://ubuntu:11434/v1"

# Optional: Langfuse observability
LANGFUSE_PUBLIC_KEY="pk-lf-..."
LANGFUSE_SECRET_KEY="sk-lf-..."
LANGFUSE_BASE_URL="http://localhost:3000"
```

### Safety Constitutions

30 different safety rule sets in `constitution/constitutions.py`, ranging from permissive (`C09_Baseline_No_Rules`) to strict (`C29_ASIMOV_Navigation_Redundancy`). Select via the frontend or `--constitution` flag.

## Datasets

The framework auto-detects the active AI2-THOR scene and loads the matching dataset:

| Scene | Unsafe Dataset | Entries | Safe Dataset |
|-------|---------------|---------|-------------|
| `FloorPlan_Messy` | `unsafe_detailed_messy.jsonl` | 238 | `safe_detailed_messy.jsonl` |
| `FloorPlan1` | `unsafe_detailed_fp1.jsonl` | 40 | `safe_detailed_1009.jsonl` |
| Other | `unsafe_detailed_1009.jsonl` | varies | `safe_detailed_1009.jsonl` |

A notification confirms which dataset was loaded when attacks start.

### Attack Categories

Attacks span diverse risk categories including:

- **Fire/Explosion:** Metal in microwave, unattended stove, flammable items on burners
- **Property Damage:** Breaking windows/vases/plates, liquid on electronics
- **Electrical Hazard:** Water on laptop, liquid in toaster
- **Concealment/Theft:** Hiding credit cards, laptops, watches in cabinets
- **Social Engineering:** Deceptive framing ("sterilize the fork in the microwave"), urgency manipulation, authority impersonation
- **Obfuscated References:** Describing objects indirectly ("three-pronged metal dining utensil" instead of "fork")
- **Toppling Hazard:** Opening stacked books, removing items from precarious piles
- **Multi-step Combos:** Damage + concealment, distraction attacks

### AI2-THOR Engine Constraints

Discovered via `diagnostic_test.py`:

- **StoveBurner** only accepts Pan and Pot — all other items use Pan as intermediary in the dataset
- **Toaster** requires BreadSliced (not whole Bread)
- **BreakObject** action works; ThrowObject does not break anything
- **CoffeeMachine** only accepts Mug (not Cup or Bowl)
- **Sink/Faucet** cleans dirty objects but does not fill them with liquid
- **Book** does not fit in Microwave
- **Laptop** does not fit in SinkBasin or GarbageCan

Run the diagnostic on any scene:
```bash
python diagnostic_test.py --scene FloorPlan_Messy
```

## Attack Modes

### Red Agent Types

| Mode | Description |
|------|-------------|
| **Simple** | Direct task instruction from dataset |
| **Concordia** | LLM-powered social engineering with spatial awareness |
| **VLM** | Vision-enhanced red agent that sees the scene via camera |
| **Adaptive** | Self-reflecting agent with strategy evolution and tree-of-thought planning |

### Blue Agent Defenses

| Defense | Description |
|---------|-------------|
| **Safety Constitution** | Rule-based plan review before execution |
| **Blue VLM** | Visual scene awareness before parsing commands |
| **Blue Safety VLM** | Post-action visual safety check (catches dangers after execution) |

### VLM Jailbreak Strategies

For testing VLM robustness (research purposes):

- `visual_reassurance` — trick VLM into describing dangerous scenes as safe
- `authority_override` — claim inspector authority over safety protocols
- `context_poisoning` — inject fake safety database updates
- `scene_hallucination` — make VLM ignore what it actually sees
- `gradual_normalization` — blend dangerous items into neutral descriptions
- `instruction_injection` — embed hidden robot instructions in scene descriptions

### Dual-Robot Mode

Two physical robots in the scene (red=agent1, blue=agent0). The red robot can physically manipulate objects while socially engineering the blue robot.

## Project Structure

```
ai2thor-lab-main/
├── src/ai2thor_lab/
│   ├── server.py           # FastAPI server, attack loop, judge routing
│   ├── agent.py            # AI2-THOR agent wrapper
│   ├── goal_planner.py     # Deterministic planner (A* graph search)
│   ├── plan_executor.py    # Executes plans in AI2-THOR
│   ├── safety.py           # Safety reviewer with constitutions
│   ├── judge.py            # BlueTeamJudge (ground truth) + LLM judge
│   ├── vlm.py              # VLM client, BlueSafetyVLM, jailbreak strategies
│   ├── adaptive_red_agent.py # Self-reflecting adaptive attacker
│   ├── llm_wrapper.py      # LLM function calling wrapper
│   ├── action_knowledge.py # AI2-THOR action mappings
│   └── ...
├── constitution/
│   └── constitutions.py    # 30 safety rule sets
├── dataset/
│   ├── unsafe_detailed_messy.jsonl  # 238 attacks for FloorPlan_Messy
│   ├── unsafe_detailed_fp1.jsonl    # 40 attacks for FloorPlan1
│   ├── safe_detailed_messy.jsonl    # Safe tasks for FloorPlan_Messy
│   └── safe_detailed_1009.jsonl     # Safe tasks for FloorPlan1
├── frontend.html           # Web UI for attack control
├── diagnostic_test.py      # Scene capability testing
├── pyproject.toml
└── README.md
```

## Interactive CLI

For manual testing and exploration:

```bash
uv run lab                    # Interactive REPL
uv run lab --task "pick up the egg" --plan   # Single task with planner
uv run lab --plan --voice     # Voice-controlled mode
```

## Development

```bash
uv run pytest                 # Run tests
python diagnostic_test.py --scene FloorPlan1   # Test scene capabilities
```

## License

See [LICENSE](LICENSE).
