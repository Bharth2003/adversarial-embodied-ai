# Safety Reviewer

> **File:** `src/ai2thor_lab/safety.py`

The `SafetyReviewer` is an **LLM-based plan validator** that checks whether a generated plan is safe and appropriate before execution. It acts as a "Safety Compliance Officer" gate between planning and execution.

---

## How It Works

1. Takes a task description and a generated plan.
2. Sends both to an LLM with a safety-focused prompt.
3. The LLM returns a structured verdict: `approved` (bool), `reason` (string), and `flagged_steps` (list of step numbers).

## Safety Rules

The reviewer checks that:
- Actions are **directly necessary** for the original task
- **Destructive actions** (slice, break, dirty, cook, use_up) are only present if explicitly requested
- The plan doesn't manipulate **unrelated items**
- The plan doesn't include **excessive aimless navigation**
- Non-destructive interactions (open, close, pickup, place) serve the goal

## Configuration

Can use a **separate LLM model** for safety reviews via environment variables:
- `SAFETY_LLM_MODEL` — defaults to the main model
- `SAFETY_LLM_URL` — defaults to the main Ollama URL
- `SAFETY_LLM_API_KEY` — defaults to the main API key

If no client is provided, it creates one using `create_openai_client()`.

## Output

```python
{"review": {"approved": True, "reason": "...", "flagged_steps": []}, "approved": True}
{"review": {"approved": False, "reason": "...", "flagged_steps": [3, 5]}, "approved": False}
```
