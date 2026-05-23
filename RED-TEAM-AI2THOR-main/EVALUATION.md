# Safety Evaluation Pipeline
This document serves as a guide for using the decoupled `generate_dataset.py` and `evaluate_safety.py` scripts to evaluate safety behavior. By splitting the pipeline into two parts, we isolate the expensive simulated physical planning from the faster evaluation LLM calls.

---

## 🏗️ Repository Structure Prerequisites
For the evaluation pipeline to function correctly, all related repositories must be cloned into the exact same parent folder (e.g., `f21ca/`), as the Python scripts rely on relative paths to locate the constitutions and the SafeAgentBench benchmark datasets.

Your directory should look like this:

```text
f21ca/
├── ai2thor-lab/      # (git@github.com:f21ca/ai2thor-lab.git)
├── constitution/     # (git@github.com:f21ca/constitution.git)
└── SafeAgentBench/   # (git@github.com:f21ca/SafeAgentBench.git)
```

Ensure you are always running these scripts from within the `ai2thor-lab` directory!

---

## 🏗️ Phase 1: Generating the Plan Dataset (`generate_dataset.py`)
This script reads human-written instructions from the `SafeAgentBench` datasets, instantiates the AI2-THOR agent, and leverages the `DeterministicPlanner` to search for guaranteed safe/unsafe action sequences. 

The output is a single dataset mapping instructions to pre-computed JSON plans alongside a ground truth `is_safe` boolean label.

### Quick Start
To generate the first 10 plans into `planned_dataset.jsonl`:
```bash
uv run python generate_dataset.py --limit 10
```

### Changing the Goal Parsing Model
`generate_dataset.py` uses an LLM to extract deterministic logic goals from natural language (e.g. `holding(Egg)`). 
To change the Goal Parser model, simply update your `.env`:

```env
ollama_model="qwen2.5-32k:latest"
ollama_url="http://ubuntu:11434/v1"
```

### Full Configuration Flags
- `--safe-file`: Path to the safe tasks JSONL. (Default: `SafeAgentBench/dataset/safe_detailed_1009.jsonl`)
- `--unsafe-file`: Path to the unsafe tasks JSONL. (Default: `SafeAgentBench/dataset/unsafe_detailed_1009.jsonl`)
- `--output`: Path to write the resulting plan dataset. (Default: `datasets/planned_dataset.jsonl`)
- `--limit`: Maximum number of tasks to process. Good for testing.

---

## 👩‍⚖️ Phase 2: Offline Safety Evaluation (`evaluate_safety.py`)
This script reads the pre-computed JSON plans generated in Phase 1 and passes them directly to the `SafetyReviewer` LLM. It operates entirely offline without spinning up the AI2-THOR unity environment. 

### Quick Start
To run evaluation using the default constitution:
```bash
uv run python evaluate_safety.py
```

### Changing the Safety Constitution
Safety rules are defined inside `../constitution/constitutions.py`. To evaluate the plans against a specific set of rules, pass the exact dictionary key to the `--constitution` argument:

```bash
# Evaluate against Anthropic 9 Rules
uv run python evaluate_safety.py --constitution C04_TopDown_Anthropic_9_Rules

# Evaluate against our custom ASIMOV Navigation rule
uv run python evaluate_safety.py --constitution C29_ASIMOV_Navigation_Redundancy
```

### Changing the Safety Evaluation Model
We deliberately decouple the model used for safety from the model used for goal generation. This allows you to test smaller, cheaper models (like `llama3:8b`) for simple goal extraction, and larger, more intelligent models (like `qwen2.5-32k`, or cloud models) as the Safety Officer.

You control the safety model explicitly via the `SAFETY_LLM_` parameters in `.env`:

```env
# Dedicated model for safety evaluations
SAFETY_LLM_MODEL="charan2:latest"
SAFETY_LLM_URL="http://ubuntu:11434/v1"
SAFETY_LLM_API_KEY="ollama"
```

### Full Configuration Flags
- `--input`: The JSONL dataset generated in Phase 1. (Default: `datasets/planned_dataset.jsonl`)
- `--output`: JSONL file to append per-task safety results into. (Default: `datasets/evaluation_results.jsonl`)
- `--report`: The final summarized spreadsheet for all instructions. (Default: `datasets/evaluation_report.csv`)
- `--summary`: A human-readable text file containing the final metrics and confusion matrix. (Default: `datasets/evaluation_summary.txt`)
- `--constitution`: Ruleset key. (Default: `C09_Baseline_No_Rules`)
- `--limit`: Number of tasks to evaluate for a quick test. 

### Understanding Output Metrics
When finished, the script calculates a confusion matrix bridging the model's approval against the dataset's ground truth string `is_safe`. This summary is printed to the console and saved to `datasets/evaluation_summary.txt`:

*   **True Positives (TP)**: Model APPROVED a SAFE plan.
*   **True Negatives (TN)**: Model REJECTED an UNSAFE plan.
*   **False Positives (FP)**: Model APPROVED an UNSAFE plan! *(Dangerous)*
*   **False Negatives (FN)**: Model REJECTED a SAFE plan! *(Annoying, Over-cautious)*

---

## 🤝 Contributing Better Safety Constitutions

The ultimate goal of this pipeline is to rigorously benchmark and improve the rules governing robot safety. If you use this script and discover that a certain constitution is too restrictive (high False Negatives) or too lenient (high False Positives), you should propose a new one!

1. Navigate to the sibling `constitution/` directory.
2. Open `constitutions.py`.
3. Add a new dictionary key with your proposed ruleset clearly articulated (e.g. `C30_Improved_Navigation_Rules`).
4. Commit your changes and push a branch to `git@github.com:f21ca/constitution.git`.
5. Open a Pull Request (PR) on GitHub detailing why your rule changes improve the benchmark scores. Include your `evaluation_report.csv` results in the PR description as proof!
