# Execution Steps — Red Team Sprint

## Pre-flight (everyone, 5 min)

```bash
# In any terminal:
ollama serve

# In another terminal:
ollama pull dolphin-llama3
ollama pull gemma3:4b

# In the repo root:
cd adversarial-embodied-safety-f21ca
cp .env.example .env
# Uncomment these lines in .env:
#   RED_AGENT_API_URL=http://localhost:11434/v1
#   RED_AGENT_MODEL_NAME=dolphin-llama3
#   BLUE_AGENT_API_URL=http://localhost:11434/v1
#   BLUE_AGENT_MODEL_NAME=gemma3:4b
```

Verify the simulation works: `python main.py --strategy leetspeak`

---

## JOHN (you) — Step by step

1. Open the repo in terminal: `cd adversarial-embodied-safety-f21ca`
2. Open Claude Code in the repo root
3. Paste the entire contents of `JOHN_CLAUDE_CODE_INSTRUCTIONS.md` as your first message
4. Claude Code will read `red/agent.py`, `red/components/base.py`, `blue/filter.py`, `blue/components/base.py`, `shared/simulation.py`, `shared/llm/model.py` first
5. It builds `evaluation/__init__.py` + `evaluation/judge.py` first (Task 1)
6. Test: `python -m evaluation.judge` — should print 2 test verdicts and a mini report
7. Tell Claude Code "now do Task 2" → builds `red/attacks/__init__.py` + `red/attacks/multi_turn.py`
8. Test: `python -m red.attacks.multi_turn` — runs 4 strategies, prints turn-by-turn results
9. "Now do Task 3" → `red/attacks/context_poison.py`
10. "Now do Task 4" → `datasets/multi_turn_cases.jsonl`
11. "Now do Task 5" → `tests/test_attacks.py`
12. Final test: `python -m tests.test_attacks`
13. Push: `git add evaluation/ red/attacks/ datasets/ tests/ && git commit -m "feat: add multi-turn attacks, judge, and integration tests" && git push`

---

## ISABEL — Step by step

1. Open Claude / ChatGPT
2. Paste **Prompt 1** from `ISABEL_PROMPTS.md` — generates `evaluation/judge.py`
   - NOTE: If John has already pushed his judge.py, skip this and use his version
3. Save output as `evaluation/judge.py` (create `evaluation/__init__.py` too)
4. Test: `python -m evaluation.judge`
5. Paste **Prompt 2** — generates `evaluation/batch_runner.py`
6. Save it. Can run partially even before John/Bharat push their attack files (graceful skip)
7. Test: `python -m evaluation.batch_runner`
8. Paste **Prompt 3** — generates `evaluation/compare_baselines.py`
9. Run: `python -m evaluation.compare_baselines`
10. Push: `git add evaluation/ && git commit -m "feat: add batch runner and baseline comparison" && git push`

---

## BHARAT — Step by step

1. Open Claude / ChatGPT
2. Paste **Prompt 1** from `BHARAT_PROMPTS.md` — generates `red/attacks/complexity_attack.py`
3. Save output (create `red/attacks/__init__.py` if it doesn't exist)
4. Test: `python -m red.attacks.complexity_attack`
5. Paste **Prompt 2** — generates `red/attacks/steganographic_attack.py`
6. Test: `python -m red.attacks.steganographic_attack`
7. Paste **Prompt 3** — generates `red/attacks/advanced_attacks.py`
8. Test: `python -m red.attacks.advanced_attacks`
9. Paste **Prompt 4** — generates `red/attacks/run_bharat_attacks.py`
10. Run: `python -m red.attacks.run_bharat_attacks` — generates `bharat_verdicts.jsonl`
11. Push: `git add red/attacks/ && git commit -m "feat: add complexity, steganographic, and advanced attacks" && git push`

---

## Merge & Final Run (last 15 min)

1. Everyone pulls latest from all branches
2. Isabel runs `python -m evaluation.batch_runner` with all attack modules present
3. This generates: `all_verdicts.jsonl`, `evaluation_report.json`
4. Isabel runs `python -m evaluation.compare_baselines`
5. This generates: `RESULTS_COMPARISON.md` with ASR numbers vs BadRobot (83%), POEX (80%), SafeAgentBench (5-10%)

---

## Key Differences from the Old .md Files

| Old (wrong) | New (correct) |
|---|---|
| Fork/microwave/metal/weapon keywords | gas/candle/flame/ignite/stove/burner/lighter/spark/fire keywords |
| `attacks/` folder at repo root | `red/attacks/` folder |
| `RedAgent` class | `create_red_agent()` function + `RedActComponent` |
| `Judge` class exists | Must be created in `evaluation/judge.py` |
| Server endpoints (POST /attack/start) | No server — direct simulation loop |
| AI2-THOR integration | Concordia framework (no AI2-THOR) |
| `src/ai2thor_lab/` folder | Does not exist |
| `llama3.1:8b` for blue | `gemma3:4b` for blue (configurable) |
| `safe_detailed_messy.jsonl` dataset | No existing datasets |
| `verdicts.jsonl` exists | Must be created by the judge |
