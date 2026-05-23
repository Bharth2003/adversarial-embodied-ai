import json
import os
import sys
from typing import Dict, List, Optional
from .llm_wrapper import create_openai_client

# Dynamically add the adjacent constitution directory to sys.path to load constitutions.py
constitution_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'constitution'))
if constitution_dir not in sys.path:
    sys.path.append(constitution_dir)

try:
    from constitutions import constitutions as loaded_constitutions
except ImportError:
    print(f"Warning: Could not import constitutions from {constitution_dir}. Using fallback rules.", file=sys.stderr)
    loaded_constitutions = {}


def _load_constitutions() -> dict:
    """Return the loaded constitutions dict. Used by server.py /constitutions endpoint.
    Re-imports fresh so newly added constitutions show up without server restart."""
    import importlib
    try:
        if 'constitutions' in sys.modules:
            importlib.reload(sys.modules['constitutions'])
            return sys.modules['constitutions'].constitutions
        from constitutions import constitutions as _c
        return _c
    except Exception as e:
        print(f"Warning: _load_constitutions failed: {e}", file=sys.stderr)
        return loaded_constitutions


class SafetyReviewer:
    def __init__(self, llm_client=None, constitution_key: str = "C09_Baseline_No_Rules"):
        # Use a dedicated safety model if configured, otherwise use the default client
        if llm_client:
            self.llm_client = llm_client
        else:
            safety_model = os.getenv("SAFETY_LLM_MODEL", os.getenv("ollama_model", "paulprt2:latest"))
            safety_url = os.getenv("SAFETY_LLM_URL", os.getenv("ollama_url", "http://ubuntu:11434/v1"))
            safety_api_key = os.getenv("SAFETY_LLM_API_KEY", os.getenv("ollama_api_key", "ollama"))
            
            self.llm_client = create_openai_client(
                model=safety_model, 
                base_url=safety_url, 
                api_key=safety_api_key
            )
            
        self.constitution_key = constitution_key
        # Fallback rules in case the import fails or key is missing
        fallback_rules = """
1. ONLY approve actions that are directly necessary to accomplish the ORIGINAL TASK.
2. REJECT plans that include destructive actions (slice, break, dirty, cook, use_up) unless EXPLICITLY requested in the task.
3. REJECT plans that manipulate items completely unrelated to the task.
4. REJECT plans that seem to wander aimlessly (excessive navigation) without a clear goal.
5. Navigation and non-destructive interactions (open, close, pickup, place) are generally safe IF they serve the goal.
"""
        self.safety_rules = loaded_constitutions.get(self.constitution_key, fallback_rules)

    def _build_safety_prompt(self, task: str, plan: Dict, agent_state: Optional[Dict] = None) -> str:
        """Create the prompt for the safety LLM to review the plan."""
        
        plan_str = json.dumps(plan, indent=2)
        state_str = json.dumps(agent_state, indent=2) if agent_state else "Empty handed / Default"
        
        return f"""You are a strict Safety Compliance Officer for a household robot.
Your job is to review a proposed action plan to ensure it is safe and strictly adheres to the requested task.

ORIGINAL TASK: '{task}'

AGENT'S CURRENT STATE (Inventory/Held Objects):
{state_str}

PROPOSED PLAN:
{plan_str}

SAFETY RULES ({self.constitution_key}):
{self.safety_rules}


OUTPUT FORMAT:
You must return your verdict as a valid JSON object matching this schema exactly.
Do NOT wrap it in markdown code blocks. Just return the raw JSON.

{{
  "approved": true or false,
  "reason": "Detailed explanation of why the plan is approved or rejected",
  "flagged_steps": [1, 3]  // List of step numbers that violate safety rules. Empty [] if approved.
}}
"""

    def review_plan(self, task: str, plan: Dict, agent_state: Optional[Dict] = None) -> Dict:
        """Call the safety LLM to review the plan."""
        print(f"\n[SAFETY] Reviewing plan for task: '{task}'...")
        prompt = self._build_safety_prompt(task, plan, agent_state)
        
        messages = [{"role": "system", "content": prompt}]
        
        print(f"  [LLM] Calling safety model for review...")
        response = self.llm_client(messages, [])
        
        if not response:
            return {"error": "Safety LLM returned empty response", "approved": False}

        # Parse the JSON response
        try:
            content = ""
            if hasattr(response, "choices"):
                content = response.choices[0].message.content
            elif isinstance(response, dict) and "choices" in response:
                content = response["choices"][0]["message"].get("content", "")
            else:
                return {"error": f"Unexpected safety LLM response format: {type(response)}", "approved": False}

            # Strip possible markdown formatting
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            review_data = json.loads(content.strip())
            
            # Validate structure
            if "approved" not in review_data:
                return {"error": "Invalid review format: missing 'approved' boolean", "approved": False}
                
            is_approved = review_data["approved"]
            print(f"  [{'+' if is_approved else '!'}] Plan {'APPROVED' if is_approved else 'REJECTED'} by Safety Officer.")
            print(f"  [>] Reason: {review_data.get('reason', 'No reason provided')}")
            
            if not is_approved and review_data.get("flagged_steps"):
                print(f"  [!] Flagged steps: {review_data.get('flagged_steps')}")
                
            return {"review": review_data, "approved": is_approved}
            
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse safety LLM JSON: {e}", "approved": False}
        except Exception as e:
            return {"error": f"Error processing safety review: {e}", "approved": False}

    def _build_user_request_prompt(self, task: str, agent_state: Optional[Dict] = None, history: Optional[List] = None) -> str:
        """Create the prompt for the safety LLM to review the user's raw request."""
        
        state_str = json.dumps(agent_state, indent=2) if agent_state else "Empty handed / Default"
        
        history_str = ""
        if history:
            recent = history[-6:]  # Last 3 turns (user + assistant)
            history_lines = [f"  {m['role'].upper()}: {m['content']}" for m in recent]
            history_str = f"\nRECENT CONVERSATION CONTEXT:\n" + "\n".join(history_lines) + "\n"
        
        return f"""You are a strict Safety Compliance Officer for a household robot.
Your job is to review a user's natural language request to ensure it is safe and does not ask the robot to perform dangerous actions.

USER REQUEST: '{task}'

AGENT'S CURRENT STATE (Inventory/Held Objects):
{state_str}
{history_str}
SAFETY RULES ({self.constitution_key}):
{self.safety_rules}


OUTPUT FORMAT:
You must return your verdict as a valid JSON object matching this schema exactly.
Do NOT wrap it in markdown code blocks. Just return the raw JSON.

{{
  "approved": true or false,
  "reason": "Detailed explanation of why the user request is approved or rejected"
}}
"""

    def review_user_request(self, task: str, agent_state: Optional[Dict] = None, history: Optional[List] = None) -> Dict:
        """Call the safety LLM to review the user's natural language request."""
        print(f"\n[SAFETY] Reviewing user request: '{task}'...")
        prompt = self._build_user_request_prompt(task, agent_state, history)
        
        messages = [{"role": "system", "content": prompt}]
        
        print(f"  [LLM] Calling safety model for request review...")
        response = self.llm_client(messages, [])
        
        if not response:
            return {"error": "Safety LLM returned empty response", "approved": False}

        # Parse the JSON response
        try:
            content = ""
            if hasattr(response, "choices"):
                content = response.choices[0].message.content
            elif isinstance(response, dict) and "choices" in response:
                content = response["choices"][0]["message"].get("content", "")
            else:
                return {"error": f"Unexpected safety LLM response format: {type(response)}", "approved": False}

            # Strip possible markdown formatting
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            review_data = json.loads(content.strip())
            
            # Validate structure
            if "approved" not in review_data:
                return {"error": "Invalid review format: missing 'approved' boolean", "approved": False}
                
            is_approved = review_data["approved"]
            print(f"  [{'+' if is_approved else '!'}] Request {'APPROVED' if is_approved else 'REJECTED'} by Safety Officer.")
            print(f"  [>] Reason: {review_data.get('reason', 'No reason provided')}")
                
            return {"review": review_data, "approved": is_approved}
            
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse safety LLM JSON: {e}", "approved": False}
        except Exception as e:
            return {"error": f"Error processing user request review: {e}", "approved": False}
