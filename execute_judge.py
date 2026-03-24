# execute_judge.py
# Run only the Judge phase (for individual execution)
# For full pipeline, use execute_pipeline.py instead

from src.agents.judge_agent import run_judge_evaluation

# Configuration — CHANGE THESE AS NEEDED
MODEL = "qwen2.5:1.5b"                           # Fast model for CPU i5
# MODEL = "qwen2.5:latest"                        # Original (slower)

run_judge_evaluation(MODEL)