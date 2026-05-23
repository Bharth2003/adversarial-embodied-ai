import json
from typing import Dict, List, Any
from .tools import get_tool_schemas, get_tool_names

class Planner:
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.tool_schemas = get_tool_schemas()
        self.tool_names = get_tool_names()

    def _build_planning_prompt(self, task: str, scene_metadata: Dict) -> str:
        """Create the prompt for the planner LLM, including all scene objects."""
        # Format the objects to be more readable
        objects_str = []
        for obj in scene_metadata.get("objects", []):
            props = []
            if obj.get("pickupable"): props.append("pickupable")
            if obj.get("receptacle"): props.append("receptacle")
            if obj.get("openable"): props.append("isOpen" if obj.get("isOpen") else "isClosed")
            if obj.get("toggleable"): props.append("isToggledOn" if obj.get("isToggled") else "isToggledOff")
            
            prop_str = f" [{', '.join(props)}]" if props else ""
            objects_str.append(f"- {obj['name']} (ID: {obj['id']}){prop_str}")

        return f"""You are an advanced planning AI for a robot in a 3D home environment (AI2-THOR simulator).

YOUR TASK: {task}
SCENE: {scene_metadata.get('scene', 'Unknown')}

Generate a step-by-step action plan to accomplish the task.
You can use the short object type name (e.g. "Pot", "Fridge", "SinkBasin", "StoveBurner") — the executor will resolve it to the correct full ID.

AVAILABLE OBJECTS:
{chr(10).join(objects_str)}

AVAILABLE TOOLS:
{", ".join(self.tool_names)}

AI2-THOR TOOL SEMANTICS (IMPORTANT — these are NOT obvious):
- navigate_to(object_id): Moves the agent near an object. Required before pickup, open, close, toggle_on, toggle_off, place_on, slice.
- pickup(object_id): Pick up an object. Must be near it. Can only hold ONE object at a time.
- place_on(receptacle_id): Place the HELD object onto a receptacle. NOTE: the argument is 'receptacle_id', not 'object_id'. Must be near the receptacle.
- fill_with_liquid(object_id, liquid): Fill a HELD object with liquid. Works ANYWHERE — you do NOT need to be near a sink. Just hold the object and call this. liquid is one of: "water", "coffee", "wine".
- cook(object_id): Cooks a HELD object. Works ANYWHERE. No need to be near anything, just be holding it.
- toggle_on/toggle_off(object_id): Turn something on/off (faucet, stoveknob). Must be near it first.
- open/close(object_id): Open/close something (fridge, cabinet). Must be near it first.
- drop(): Drop the held object.
- finish(): End the task.

PLANNING RULES:
1. navigate_to the target object BEFORE pickup, open, close, toggle, slice, place_on.
2. You can only hold ONE object. To pick up another, place_on or drop first.
3. fill_with_liquid and cook work on HELD objects and do NOT require navigation.
4. To place something on a stove burner, navigate to the burner FIRST, then place_on.
5. Plan must end with finish().

OUTPUT FORMAT:
Return a valid JSON object. Do NOT wrap in markdown code blocks.

{{
  "task": "{task}",
  "steps": [
    {{"step": 1, "tool": "navigate_to", "arguments": {{"object_id": "Pot"}}, "reason": "Go to the pot"}},
    {{"step": 2, "tool": "pickup", "arguments": {{"object_id": "Pot"}}, "reason": "Pick it up"}},
    {{"step": 3, "tool": "fill_with_liquid", "arguments": {{"object_id": "Pot", "liquid": "water"}}, "reason": "Fill with water (works anywhere while holding)"}},
    {{"step": 4, "tool": "navigate_to", "arguments": {{"object_id": "StoveBurner"}}, "reason": "Go to stove"}},
    {{"step": 5, "tool": "place_on", "arguments": {{"receptacle_id": "StoveBurner"}}, "reason": "Place pot on burner"}},
    {{"step": 6, "tool": "navigate_to", "arguments": {{"object_id": "StoveKnob"}}, "reason": "Go to knob"}},
    {{"step": 7, "tool": "toggle_on", "arguments": {{"object_id": "StoveKnob"}}, "reason": "Turn on burner"}},
    {{"step": 8, "tool": "finish", "arguments": {{}}, "reason": "Done"}}
  ]
}}
"""

    def generate_plan(self, task: str, scene_metadata: Dict) -> Dict:
        """Call the LLM to generate a plan based on the task and scene metadata."""
        print(f"\n[PLANNER] Generating plan for task: '{task}'...")
        prompt = self._build_planning_prompt(task, scene_metadata)
        
        messages = [{"role": "system", "content": prompt}]
        
        print(f"  [LLM] Calling model to generate plan...")
        # Call the LLM (no tools needed since we are just asking for a JSON response)
        response = self.llm_client(messages, [])
        
        if not response:
            return {"error": "LLM returned empty response", "success": False}

        # Parse the JSON response
        try:
            content = ""
            # Handle OpenAI SDK response
            if hasattr(response, "choices"):
                content = response.choices[0].message.content
            # Handle dictionary block response
            elif isinstance(response, dict) and "choices" in response:
                content = response["choices"][0]["message"].get("content", "")
            else:
                return {"error": f"Unexpected LLM response format: {type(response)}", "success": False}

            # Strip possible markdown formatting
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
                
            plan_data = json.loads(content.strip())
            
            # Validate structure
            if "steps" not in plan_data:
                return {"error": "Invalid plan format: missing 'steps' array", "success": False}
                
            print(f"  [+] Plan generated successfully with {len(plan_data['steps'])} steps.")
            return {"plan": plan_data, "success": True}
            
        except json.JSONDecodeError as e:
            return {"error": f"Failed to parse LLM JSON: {e}\nRaw output: {content[:100]}...", "success": False}
        except Exception as e:
            return {"error": f"Error processing plan: {e}", "success": False}

    def print_plan(self, plan: Dict):
        """Pretty print the generated plan."""
        print("\n" + "="*50)
        print("📝 ACTION PLAN")
        print("="*50)
        for step in plan.get("steps", []):
            tool = step.get("tool")
            args = step.get("arguments", {})
            reason = step.get("reason", "No reason provided")
            
            arg_str = ", ".join(f"{k}={v.split('|')[0] if isinstance(v, str) and '|' in v else v}" for k, v in args.items())
            print(f"[{step.get('step')}] {tool}({arg_str})")
            print(f"    └─ {reason}")
        print("="*50 + "\n")
