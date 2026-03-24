# execute_red.py
# Run only the Red Agent phase (for individual execution)
# For full pipeline, use execute_pipeline.py instead

from src.agents.red_agent import run_red_generation
from src.prompts.strategies import get_top_strategies

# Configuration — CHANGE THESE AS NEEDED
MODEL = "qwen2.5:1.5b"                           # Fast model for CPU i5
# MODEL = "dolphin-llama3:latest"                 # Original (slower but stronger)
# MODEL = "red-agent-qwen"                        # Fine-tuned model (after training)

DATA_PATH = "data/safeagentbench/unsafe_detailed_1009.jsonl"
STRATEGIES = get_top_strategies(6)                 # Top 6 most effective strategies
MAX_SEEDS = 30                                     # Reduced from 50 for speed
K_CANDIDATES = 1                                   # Reduced from 2 for speed

run_red_generation(DATA_PATH, MODEL, STRATEGIES, K_CANDIDATES, MAX_SEEDS)