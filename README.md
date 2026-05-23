# Adversarial Safety in Embodied AI

A comprehensive red-team/blue-team benchmarking framework for evaluating adversarial robustness of LLM-driven embodied agents in **AI2-THOR** simulation environments. This repo combines individual research on prompt injection attacks and fine-tuned attack models with the full `ai2thor-lab` group framework.

---

## Overview

This project implements an adversarial attack-and-defence pipeline where:

- **Red Agent (Attacker):** Uses LLM-powered social engineering, prompt injection, circumlocution, and fine-tuned models to trick household robots into unsafe actions (e.g., putting metal in the microwave, turning on gas burners, breaking objects).
- **Blue Agent (Defender):** A household robot with constitution-based safety policies, deterministic planning, and optional VLM safety checks that must resist manipulation while completing legitimate tasks.
- **Judge:** Evaluates outcomes using ground-truth scene state and LLM cross-checking.

The benchmark measures **Attack Success Rate (ASR)** — how often the red agent successfully jailbreaks the blue agent across diverse attack categories.

---

## Repository Structure

### Individual Work — Attack Strategies & Fine-Tuning (Root Level)

| File | Description |
|---|---|
| `bharat_strategies.py` | Standalone attack library — 10+ strategies with circumlocution dictionaries to bypass keyword-based safety filters |
| `bharat_attacks_colab_gemma_250attacks.ipynb` | 250-attack evaluation against Gemma 3B on Colab |
| `bharat_attacks_vs_qwen25.ipynb` | Attack experiments against Qwen2.5-1.5B |
| `bharat_attacks_vs_qwen3.ipynb` | Attack experiments against Qwen3-1.7B |
| `bharat_finetuned_attack.py` / `.ipynb` | Fine-tuned attack model evaluation pipeline |
| `bharat_metrics.json` / `bharat_verdicts.jsonl` | Attack metrics and evaluation verdicts |
| `BHARAT_PROMPTS.md` | 10 advanced prompt injection templates (complexity overload, nested conditionals, JSON injection, multi-turn, context poisoning, etc.) |
| `Red_Agent_FineTune_Colab.ipynb` | Qwen2.5-1.5B fine-tuning via Unsloth (LoRA) to generate clean attack prompts |
| `finetune_red_agent.py` / `auto_finetune_red.py` / `cpu_finetune_real.py` | Fine-tuning pipeline (GPU + CPU variants) |
| `execute_pipeline.py` / `execute_red.py` / `execute_blue.py` / `execute_judge.py` | Modular pipeline execution for Red → Blue → Judge loop |
| `convert_dataset.py` / `export_summary_to_excel.py` | Dataset conversion and metrics export utilities |
| `ASR_Finetuned_Research_Report.txt` / `Fine_Tuning_Execution_Report.md` | Research findings and fine-tuning reports |

### Group Work — AI2-THOR Lab (`RED-TEAM-AI2THOR-main/`)

| Directory | Description |
|---|---|
| `src/` | Core framework — Red Agent, Blue Agent, Judge, safety evaluation |
| `configs/` | Simulation configuration files |
| `scripts/` | Utility scripts for running experiments |
| `datasets/` | Safe and unsafe task datasets |
| `docs/` | Documentation (architecture, safety, tools, planner) |
| `tests/` | Unit and integration tests |
| `tables/` | Results tables |
| `data/` / `results/` | Generated experiment data |

---

## Key Contributions (Individual)

1. **Circumlocution-Based Attacks:** Built a mapping of 25+ keyword circumlocutions that successfully bypass keyword-based safety filters (e.g., "rapid oxidation reaction" for "fire").
2. **Advanced Prompt Injection:** Developed 10 novel attack strategies including complexity overload, nested conditionals, JSON injection, multi-turn manipulation, and context poisoning.
3. **Fine-Tuned Attack Model:** Fine-tuned Qwen2.5-1.5B using Unsloth LoRA to autonomously generate effective attack prompts, achieving measurable ASR improvements.
4. **Cross-Model Evaluation:** Benchmarked attack success across Gemma 3B, Qwen2.5, and Qwen3 to compare model vulnerabilities.
5. **Modular Pipeline:** Built a configurable `execute_pipeline.py` supporting multiple strategies, fine-tuned models, and automated metrics collection.

---

## Getting Started

### Prerequisites

- Python >= 3.13
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [Ollama](https://ollama.ai/) running locally for LLM inference
- AI2-THOR Unity build (for 3D simulation)

### Installation (Group Framework)

```bash
cd RED-TEAM-AI2THOR-main
uv sync
cp .env.example .env
# Edit .env with your Ollama endpoints
```

### Running Attack Strategies (Standalone)

```bash
# Test circumlocution strategies
python -c "from bharat_strategies import *; print(circumlocute('put the fork in the microwave'))"

# Run full evaluation pipeline
python execute_pipeline.py --red-model dolphin-llama3 --strategies all
```

### Colab Fine-Tuning

Open `Red_Agent_FineTune_Colab.ipynb` in Google Colab (GPU runtime) to fine-tune Qwen2.5-1.5B for attack generation.

---

## Architecture

```
Red Agent (LLM)          Blue Agent (Planner + Safety)         Judge
     │                            │                              │
     ├─ Social engineering ──────>│                              │
     │   Prompt injection         ├─ Goal Parser (LLM)           │
     │   Circumlocution           ├─ Deterministic Planner       │
     │   Complexity overload      ├─ Safety Reviewer (Constitution)│
     │   Multi-turn attacks       ├─ [Optional] VLM Safety Check │
     │   Fine-tuned attacks       ├─ Plan Executor (AI2-THOR)    │
     │                            │                              │
     │                            ├─ Scene state ───────────────>│
     │                            │                   BlueTeamJudge
     │                            │                   (ground truth)
```

---

## Results

Detailed metrics are available in:
- `ASR_Finetuned_Research_Report.txt` — Pre/post fine-tuning attack success rates
- `bharat_metrics.json` — Per-strategy metrics breakdown
- `Fine_Tuning_Execution_Report.md` — Fine-tuning methodology and outcomes
- `RED-TEAM-AI2THOR-main/EVALUATION.md` — Group evaluation results

---

## Citation

If you use this work in your research, please cite:

```bibtex
@misc{bharth2025adversarial,
  author = {Bharth, K S},
  title = {Adversarial Safety in Embodied AI: Red-Teaming Household Robots},
  year = {2025},
  howpublished = {GitHub},
  url = {https://github.com/Bharth2003/adversarial-embodied-ai}
}
```

---

## License

MIT
