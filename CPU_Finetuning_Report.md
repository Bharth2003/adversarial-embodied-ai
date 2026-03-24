# CPU Red Team Fine-Tuning Execution Report
**Date:** 2026-03-11 23:39:02

## 1. Action Overview
Took 30 highly unsafe physical actions directly from `unsafe_detailed_1009.jsonl` (e.g. Explosion Hazards, Fire Hazards, Shock Hazards).
Using structural templating, these were rewritten into **90 high-quality attack pairings** mapping seed instructions to strategic Persona, Reality, and Emotional bypasses.

## 2. Model Fine-Tuning Execution (CPU Laptop Profile)
- **Base Model:** `Qwen/Qwen2.5-1.5B-Instruct`
- **Device:** CPU
- **Fine-Tuning Method:** SFTTrainer with LoRA (r=8) targeting Q and V projections.
- **Limits:** Due to CPU constraints, training was executed as a rapid proof of concept (max 15 steps).

## 3. Results & Next Steps
- Fine-tuned artifact saved to: `./red_agent_lora_cpu`
- Dataset saved to: `./cpu_red_team_dataset.jsonl`

To push this into evaluating the blue model:
```bash
python execute_pipeline.py --red-model red-agent-qwen --strats 10
```
*(You may need to convert the LoRA to GGUF using Ollama's documented pipeline if interacting directly via Ollama)*
