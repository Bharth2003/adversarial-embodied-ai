# src/agents/red_agent.py
# Enhanced Red Team Agent — Optimized for CPU speed and high ASR
# Uses advanced multi-strategy attack generation with CoT reasoning

import pandas as pd
import time
from tqdm import tqdm
from src.ollama_client import ollama_generate
from src.prompts.strategies import build_red_prompt, get_top_strategies

def run_red_generation(data_path, model_name, strategies=None, k_candidates=1, max_seeds=30):
    """
    Generate red team attack prompts against the blue agent.
    
    Optimizations for CPU i5:
    - Reduced max_seeds from 50 → 30 (configurable)
    - Use top strategies only (default 6 instead of 12) 
    - k_candidates=1 (was 2)
    - num_predict=200 (shorter outputs)
    - Saves progress after each seed to avoid lost work
    
    Args:
        data_path: Path to the unsafe_detailed_1009.jsonl dataset
        model_name: Ollama model name for red agent
        strategies: List of strategy IDs to use (default: top 6)
        k_candidates: Number of candidates per strategy (default: 1)
        max_seeds: Maximum number of seeds to process (default: 30)
    """
    # Use top strategies if none specified
    if strategies is None:
        strategies = get_top_strategies(6)
    
    # Load SafeAgentBench data
    seeds = pd.read_json(data_path, lines=True).head(max_seeds)
    results = []
    total_calls = len(seeds) * len(strategies) * k_candidates
    
    print(f"--- RED TEAM PHASE ---")
    print(f"  Model: {model_name}")
    print(f"  Seeds: {len(seeds)} | Strategies: {len(strategies)} | K: {k_candidates}")
    print(f"  Total LLM calls: {total_calls}")
    print(f"  Estimated time: ~{total_calls * 5 // 60} minutes (at ~5s/call on CPU)")
    print()
    
    start_time = time.time()
    call_count = 0
    
    for idx, item in tqdm(seeds.iterrows(), total=len(seeds), desc="Red Agent"):
        # Extract the seed text from the dataset
        seed_text = item.get("instruction") or item.get("prompt") or item.get("task")
        if not seed_text:
            continue
            
        # Extract metadata
        task_id = item.get("id") or item.get("task_id") or idx
        risk_category = item.get("risk_category", "unknown")
        
        for strategy in strategies:
            for k in range(k_candidates):
                call_count += 1
                
                # Build the attack prompt with CoT reasoning
                prompt = build_red_prompt(seed_text, strategy)
                
                # Generate with the red model — shorter output for speed
                out = ollama_generate(
                    model_name, 
                    prompt, 
                    temperature=0.85,     # Slightly higher temp for creativity
                    num_predict=200       # Reduced from 400 for speed
                )
                
                # Extract the spoken command from <answer> tags
                try:
                    test_prompt = out.split("<answer>")[1].split("</answer>")[0].strip()
                except (IndexError, AttributeError):
                    # If no tags, use the raw output but clean it up
                    test_prompt = out.strip()
                    # Remove any thinking sections if present
                    if "<think>" in test_prompt:
                        try:
                            test_prompt = test_prompt.split("</think>")[1].strip()
                        except IndexError:
                            pass
                
                # Skip empty outputs
                if not test_prompt:
                    continue
                
                results.append({
                    "task_id": task_id,
                    "seed_instruction": seed_text,
                    "risk_category": risk_category,
                    "strategy": strategy,
                    "k": k,
                    "test_prompt": test_prompt,
                    "red_model": model_name
                })
        
        # Save progress after each seed (crash recovery)
        if (idx + 1) % 5 == 0:
            pd.DataFrame(results).to_csv("data/interim_red_outputs.csv", index=False)
    
    # Final save
    df = pd.DataFrame(results)
    df.to_csv("data/interim_red_outputs.csv", index=False)
    
    elapsed = time.time() - start_time
    print(f"\n--- Red Phase Complete ---")
    print(f"  Generated {len(results)} attack prompts in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Output saved to data/interim_red_outputs.csv")
    
    return df