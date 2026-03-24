# execute_pipeline.py
# ============================================================
# UNIFIED FAST PIPELINE — Red → Blue → Judge in one script
# Optimized for CPU i5 to run in ~20 minutes
# ============================================================
#
# Usage:
#   python execute_pipeline.py                          # Default: 30 seeds, 6 strategies
#   python execute_pipeline.py --seeds 50 --strats 8    # Custom: 50 seeds, 8 strategies
#   python execute_pipeline.py --fast                   # Ultra-fast: 15 seeds, 4 strategies
#
# Models (change these to match what you have in Ollama):
#   Red:   qwen2.5:1.5b  (or dolphin-llama3:latest for better attacks)
#   Blue:  qwen2.5:1.5b  (or gemma3:4b — but slower)
#   Judge: qwen2.5:1.5b  (or qwen2.5:latest — but slower)
# ============================================================

import sys
import time
import argparse
from src.agents.red_agent import run_red_generation
from src.agents.blue_agent import run_blue_response
from src.agents.judge_agent import run_judge_evaluation
from src.prompts.strategies import get_top_strategies

def main():
    parser = argparse.ArgumentParser(description="Red vs Blue Team Pipeline")
    parser.add_argument("--seeds", type=int, default=30, help="Number of seeds to process")
    parser.add_argument("--strats", type=int, default=6, help="Number of strategies to use")
    parser.add_argument("--k", type=int, default=1, help="Candidates per strategy")
    parser.add_argument("--fast", action="store_true", help="Ultra-fast mode (15 seeds, 4 strategies)")
    parser.add_argument("--red-model", type=str, default="qwen2.5:1.5b", help="Red agent model")
    parser.add_argument("--blue-model", type=str, default="qwen2.5:1.5b", help="Blue agent model")
    parser.add_argument("--judge-model", type=str, default="qwen2.5:1.5b", help="Judge model")
    args = parser.parse_args()
    
    # Ultra-fast mode overrides
    if args.fast:
        args.seeds = 15
        args.strats = 4
        args.k = 1
    
    # Configuration
    DATA_PATH = "data/safeagentbench/unsafe_detailed_1009.jsonl"
    strategies = get_top_strategies(args.strats)
    
    total_calls = args.seeds * len(strategies) * args.k * 3  # ×3 for red+blue+judge
    
    print("=" * 60)
    print("  RED vs BLUE TEAM — UNIFIED PIPELINE")
    print("=" * 60)
    print(f"  Red Model:    {args.red_model}")
    print(f"  Blue Model:   {args.blue_model}")
    print(f"  Judge Model:  {args.judge_model}")
    print(f"  Seeds:        {args.seeds}")
    print(f"  Strategies:   {strategies}")
    print(f"  K-candidates: {args.k}")
    print(f"  Total LLM calls: ~{total_calls}")
    print(f"  Est. time: ~{total_calls * 4 // 60} minutes")
    print("=" * 60)
    print()
    
    pipeline_start = time.time()
    
    # ═══════════════════════════════════════════
    # PHASE 1: RED AGENT — Generate attack prompts
    # ═══════════════════════════════════════════
    print("\n" + "━" * 50)
    print("  PHASE 1/3: RED AGENT")
    print("━" * 50)
    
    run_red_generation(
        data_path=DATA_PATH,
        model_name=args.red_model,
        strategies=strategies,
        k_candidates=args.k,
        max_seeds=args.seeds
    )
    
    # ═══════════════════════════════════════════
    # PHASE 2: BLUE AGENT — Generate responses
    # ═══════════════════════════════════════════
    print("\n" + "━" * 50)
    print("  PHASE 2/3: BLUE AGENT")
    print("━" * 50)
    
    run_blue_response(
        model_name=args.blue_model,
        input_csv="data/interim_red_outputs.csv",
        output_csv="data/interim_blue_responses.csv"
    )
    
    # ═══════════════════════════════════════════
    # PHASE 3: JUDGE — Evaluate compliance
    # ═══════════════════════════════════════════
    print("\n" + "━" * 50)
    print("  PHASE 3/3: JUDGE")
    print("━" * 50)
    
    run_judge_evaluation(
        model_name=args.judge_model,
        input_csv="data/interim_blue_responses.csv",
        output_csv="results/final_redteam_results.csv"
    )
    
    # ═══════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════
    total_time = time.time() - pipeline_start
    
    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time: {total_time:.0f}s ({total_time/60:.1f} minutes)")
    print(f"  Results: results/final_redteam_results.csv")
    print(f"  Report:  results/ASR_Research_Report.xlsx")
    print("=" * 60)


if __name__ == "__main__":
    main()
