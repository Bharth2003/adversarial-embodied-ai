# src/eval/run_redteam_eval.py
import pandas as pd
from tqdm import tqdm

def main():
    # Model Setup
    red_cfg = RedConfig(model="dolphin-llama3:latest")
    blue_cfg = BlueConfig(model="gemma3:4b")
    jdg_cfg = JudgeConfig(model="qwen2.5:latest")
    
    # Load dataset
    seeds = load_seeds("data/safeagentbench/unsafe_detailed_1009.jsonl", limit=100)
    K = 4 # Reduced for local stability 
    all_data = []

    # --- PHASE 1: GENERATE ALL ATTACKS (RED MODEL ONLY) ---
    # This prevents model swapping by finishing all attacker turns first 
    print(f"Phase 1: Generating attacks using {red_cfg.model}...")
    for item in tqdm(seeds):
        seed_text = item.get("prompt") or item.get("instruction") or item.get("task")
        if not seed_text: continue
        for strategy in STRATEGIES:
            for k in range(K):
                test = generate_test_prompt(red_cfg, seed_text, strategy_id=strategy)
                all_data.append({
                    "seed_id": item.get("id") or item.get("task_id"),
                    "strategy": strategy,
                    "k": k,
                    "red_model": red_cfg.model,
                    "test_prompt": test["test_prompt"]
                })

    # --- PHASE 2: GENERATE ALL RESPONSES (BLUE MODEL ONLY) ---
    print(f"Phase 2: Generating responses using {blue_cfg.model}...")
    for row in tqdm(all_data):
        row["blue_model"] = blue_cfg.model
        row["blue_out"] = answer_blue(blue_cfg, row["test_prompt"])

    # --- PHASE 3: EVALUATE ALL (JUDGE MODEL ONLY) ---
    print(f"Phase 3: Judging safety using {jdg_cfg.model}...")
    for row in tqdm(all_data):
        verdict = judge(jdg_cfg, row["test_prompt"], row["blue_out"])
        row.update({
            "judge_model": jdg_cfg.model,
            "refused": verdict.get("refused"),
            "unsafe_compliance": verdict.get("unsafe_compliance"),
            "over_refusal": verdict.get("over_refusal"),
            "notes": verdict.get("notes")
        })

    # --- SAVE AND LOG ---
    df = pd.DataFrame(all_data)
    df.to_csv("results/redteam_item_level.csv", index=False)
    
    summary = df.groupby(["red_model", "blue_model", "strategy"]).agg(
        n=("unsafe_compliance", "count"),
        ASR_pct=("unsafe_compliance", lambda x: 100.0 * (x == True).mean()),
        Refusal_pct=("refused", lambda x: 100.0 * (x == True).mean())
    ).reset_index()
    
    summary.to_csv("results/redteam_summary.csv", index=False)
    print("\nSimulation Complete. Results saved to results/ folder.")
    print(summary.to_string(index=False))