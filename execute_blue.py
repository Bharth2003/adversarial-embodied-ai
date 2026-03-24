# execute_blue.py
# Run only the Blue Agent phase (for individual execution)
# For full pipeline, use execute_pipeline.py instead

from src.agents.blue_agent import run_blue_response

# Configuration — CHANGE THESE AS NEEDED
MODEL = "qwen2.5:1.5b"                           # Fast model for CPU i5
# MODEL = "gemma3:4b"                             # Original (slower, more safety-aligned)

run_blue_response(MODEL)