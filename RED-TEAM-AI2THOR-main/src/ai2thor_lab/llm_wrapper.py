import json
from typing import List, Dict, Optional, Any, Callable
from .agent import Agent
from .navigator import Navigator
from .tools import get_tool_schemas, get_tool_names
import os
import uuid
from dotenv import load_dotenv

load_dotenv()

class LLMWrapper:
    def __init__(self, agent: Agent, llm_client: Optional[Any] = None):
        self.agent = agent
        self.navigator = Navigator(agent.controller)
        self.llm_client = llm_client
        self.conversation_history: List[Dict[str, str]] = []
        self.tool_schemas = get_tool_schemas()
        self.tool_names = get_tool_names()
        self.max_iterations = 50
        

    def set_llm_client(self, client: Callable):
        """Set the LLM client function (e.g., OpenAI, Anthropic)"""
        self.llm_client = client

    def _format_system_prompt(self, goal: str) -> str:
        """Format the system prompt with goal and available tools"""
        tools_list = ", ".join(self.tool_names)
        return f"""You are an AI agent controlling a robot in a 3D home environment (AI2-THOR).

GOAL: {goal}

AVAILABLE TOOLS:
{tools_list}

IMPORTANT NOTES:
- ALWAYS call 'get_observation' first to see what objects are visible
- Use 'get_object_metadata' to interrogate specific objects before interacting with them (e.g. to check if a receptacle is open, what it contains, or if it is pickupable)
- To interact with a distant object: first 'navigate_to' it, THEN interact
- Use 'navigate_to' to move to objects - do NOT manually issue multiple MoveAhead commands
- After navigation, check if you're close enough to interact
- Object IDs look like 'Fridge|-02.10|+00.00|+01.07' — use EXACT IDs from observations
- After completing ALL sub-goals, call 'finish' with a summary message
- If an action fails, note the error and try a different approach

Respond with a JSON object containing 'tool' and 'arguments' fields."""

    def _format_messages(self, goal: str) -> List[Dict[str, str]]:
        """Format messages for LLM API call"""
        messages = [{"role": "system", "content": self._format_system_prompt(goal)}]
        messages.extend(self.conversation_history)
        return messages

    def _call_llm(self, goal: str) -> Optional[Dict]:
        """Call the LLM to get the next action
        
        Handles both:
        - OpenAI SDK ChatCompletion objects (from create_openai_client)
        """
        if not self.llm_client:
            raise ValueError("LLM client not set. Use set_llm_client() to configure.")

        messages = self._format_messages(goal)
        response = self.llm_client(messages, self.tool_schemas)
        if not response:
            return None

        # --- Handle OpenAI SDK ChatCompletion object ---
        if hasattr(response, "choices"):
            choice = response.choices[0]
            msg = choice.message
            # Try tool_calls first (function-calling mode)
            if msg.tool_calls:
                tc = msg.tool_calls[0]
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                return {"tool": tc.function.name, "arguments": args}
            # Fall back to content
            content = msg.content or ""
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"tool": "get_observation", "arguments": {}}
            return None

        # --- Handle dict responses (ollama-style) ---
        if isinstance(response, dict) and "choices" in response:
            content = response["choices"][0]["message"].get("content", "")
            if content:
                try:
                    return json.loads(content)
                except json.JSONDecodeError:
                    return {"tool": "get_observation", "arguments": {}}

        return None

    def _execute_tool(self, tool_name: str, arguments: Dict) -> Dict:
        """Execute a tool by calling the AI2-THOR controller directly with exact object IDs."""
        result = {"success": False, "message": ""}

        try:
            # --- Navigation ---
            if tool_name == "navigate_to":
                success, msg = self.navigator.navigate_to_object(
                    self.agent, arguments["object_id"]
                )
                result = {"success": success, "message": msg}

            # --- Direct controller actions with exact object IDs ---
            elif tool_name == "pickup":
                event = self.agent.controller.step(
                    action="PickupObject", objectId=arguments["object_id"]
                )
                success = event.metadata["lastActionSuccess"]
                if success:
                    self.agent.held_object = arguments["object_id"]
                    print(f"  [HOLD] Now holding: {arguments['object_id'].split('|')[0]}")
                result = {
                    "success": success,
                    "message": "Picked up" if success else event.metadata.get("errorMessage", "Failed to pick up"),
                }

            elif tool_name == "drop":
                event = self.agent.controller.step(action="DropHandObject")
                success = event.metadata["lastActionSuccess"]
                if success and self.agent.held_object:
                    print(f"  [DROP] Released: {(self.agent.held_object or '').split('|')[0]}")
                    self.agent.held_object = None
                result = {
                    "success": success,
                    "message": "Dropped" if success else event.metadata.get("errorMessage", "Failed to drop"),
                }

            elif tool_name == "place_on":
                event = self.agent.controller.step(
                    action="PutObject",
                    objectId=arguments["receptacle_id"],
                    forceAction=False,
                    placeStationary=True,
                )
                success = event.metadata["lastActionSuccess"]
                if success and self.agent.held_object:
                    print(f"  [PLACE] Placed {(self.agent.held_object or '').split('|')[0]} on {arguments['receptacle_id'].split('|')[0]}")
                    self.agent.held_object = None
                result = {
                    "success": success,
                    "message": "Placed" if success else event.metadata.get("errorMessage", "Failed to place"),
                }

            elif tool_name == "open":
                event = self.agent.controller.step(
                    action="OpenObject", objectId=arguments["object_id"]
                )
                success = event.metadata["lastActionSuccess"]
                result = {
                    "success": success,
                    "message": "Opened" if success else event.metadata.get("errorMessage", "Failed to open"),
                }

            elif tool_name == "close":
                event = self.agent.controller.step(
                    action="CloseObject", objectId=arguments["object_id"]
                )
                success = event.metadata["lastActionSuccess"]
                result = {
                    "success": success,
                    "message": "Closed" if success else event.metadata.get("errorMessage", "Failed to close"),
                }

            elif tool_name == "toggle_on":
                event = self.agent.controller.step(
                    action="ToggleObjectOn", objectId=arguments["object_id"]
                )
                success = event.metadata["lastActionSuccess"]
                result = {
                    "success": success,
                    "message": "Turned on" if success else event.metadata.get("errorMessage", "Failed to turn on"),
                }

            elif tool_name == "toggle_off":
                event = self.agent.controller.step(
                    action="ToggleObjectOff", objectId=arguments["object_id"]
                )
                success = event.metadata["lastActionSuccess"]
                result = {
                    "success": success,
                    "message": "Turned off" if success else event.metadata.get("errorMessage", "Failed to turn off"),
                }

            elif tool_name == "slice":
                event = self.agent.controller.step(
                    action="SliceObject", objectId=arguments["object_id"]
                )
                success = event.metadata["lastActionSuccess"]
                if success and self.agent.held_object == arguments.get("object_id"):
                    self.agent.held_object = None
                result = {
                    "success": success,
                    "message": "Sliced" if success else event.metadata.get("errorMessage", "Failed to slice"),
                }

            elif tool_name == "cook":
                event = self.agent.controller.step(
                    action="CookObject", objectId=arguments["object_id"]
                )
                success = event.metadata["lastActionSuccess"]
                result = {
                    "success": success,
                    "message": "Cooked" if success else event.metadata.get("errorMessage", "Failed to cook"),
                }

            elif tool_name == "fill_with_liquid":
                event = self.agent.controller.step(
                    action="FillObjectWithLiquid",
                    objectId=arguments["object_id"],
                    fillLiquid=arguments["liquid"],
                )
                success = event.metadata["lastActionSuccess"]
                result = {
                    "success": success,
                    "message": "Filled" if success else event.metadata.get("errorMessage", "Failed to fill"),
                }

            elif tool_name == "empty_liquid":
                event = self.agent.controller.step(
                    action="EmptyLiquidFromObject", objectId=arguments["object_id"]
                )
                success = event.metadata["lastActionSuccess"]
                result = {
                    "success": success,
                    "message": "Emptied" if success else event.metadata.get("errorMessage", "Failed to empty"),
                }

            elif tool_name == "clean":
                event = self.agent.controller.step(
                    action="CleanObject", objectId=arguments["object_id"]
                )
                success = event.metadata["lastActionSuccess"]
                result = {
                    "success": success,
                    "message": "Cleaned" if success else event.metadata.get("errorMessage", "Failed to clean"),
                }

            elif tool_name == "dirty":
                event = self.agent.controller.step(
                    action="DirtyObject", objectId=arguments["object_id"]
                )
                success = event.metadata["lastActionSuccess"]
                result = {
                    "success": success,
                    "message": "Dirtied" if success else event.metadata.get("errorMessage", "Failed to dirty"),
                }

            elif tool_name == "break_object":
                event = self.agent.controller.step(
                    action="BreakObject", objectId=arguments["object_id"]
                )
                success = event.metadata["lastActionSuccess"]
                if success and self.agent.held_object == arguments.get("object_id"):
                    self.agent.held_object = None
                result = {
                    "success": success,
                    "message": "Broken" if success else event.metadata.get("errorMessage", "Failed to break"),
                }

            elif tool_name == "use_up":
                event = self.agent.controller.step(
                    action="UseUpObject", objectId=arguments["object_id"]
                )
                success = event.metadata["lastActionSuccess"]
                result = {
                    "success": success,
                    "message": "Used up" if success else event.metadata.get("errorMessage", "Failed to use up"),
                }

            elif tool_name == "finish":
                result = {
                    "success": True,
                    "message": arguments.get("message", "Task completed"),
                }

            # --- Movement/rotation (still go through execute_command) ---
            elif tool_name == "look_up":
                success = self.agent.execute_command("look up")
                result = {"success": success, "message": "Looked up" if success else "Failed"}

            elif tool_name == "look_down":
                success = self.agent.execute_command("look down")
                result = {"success": success, "message": "Looked down" if success else "Failed"}

            elif tool_name == "look_straight":
                success = self.agent.execute_command("look straight")
                result = {"success": success, "message": "Looked straight" if success else "Failed"}

            elif tool_name == "rotate_left":
                degrees = arguments.get("degrees", 15)
                success = self.agent.execute_command(f"rotate left {degrees}")
                result = {"success": success, "message": "Rotated left" if success else "Failed"}

            elif tool_name == "rotate_right":
                degrees = arguments.get("degrees", 15)
                success = self.agent.execute_command(f"rotate right {degrees}")
                result = {"success": success, "message": "Rotated right" if success else "Failed"}

            elif tool_name == "move_forward":
                distance = arguments.get("distance", 0.1)
                success = self.agent.execute_command(f"forward {distance}")
                result = {"success": success, "message": "Moved forward" if success else "Failed"}

            elif tool_name == "move_back":
                distance = arguments.get("distance", 0.1)
                success = self.agent.execute_command(f"back {distance}")
                result = {"success": success, "message": "Moved back" if success else "Failed"}

            elif tool_name == "move_left":
                distance = arguments.get("distance", 0.1)
                success = self.agent.execute_command(f"left {distance}")
                result = {"success": success, "message": "Moved left" if success else "Failed"}

            elif tool_name == "move_right":
                distance = arguments.get("distance", 0.1)
                success = self.agent.execute_command(f"right {distance}")
                result = {"success": success, "message": "Moved right" if success else "Failed"}

            elif tool_name == "crouch":
                success = self.agent.execute_command("crouch")
                result = {"success": success, "message": "Crouched" if success else "Failed"}

            elif tool_name == "stand":
                success = self.agent.execute_command("stand")
                result = {"success": success, "message": "Stood up" if success else "Failed"}

            # --- Observation tools ---
            elif tool_name == "get_observation":
                observation = self.agent.get_context_description()
                result = {"success": True, "message": observation}

            elif tool_name == "list_objects":
                objects = self.agent.get_visible_objects()
                obj_list = "\n".join(
                    [
                        f"- {o['name']} (ID: {o['id']}) at {o['distance']:.1f}m"
                        for o in objects
                    ]
                )
                result = {"success": True, "message": obj_list}

            elif tool_name == "get_position":
                pos = self.agent.get_agent_position()
                result = {"success": True, "message": f"Position: {pos}"}

            elif tool_name == "get_object_metadata":
                obj_id = arguments.get("object_id")
                objects = self.agent.controller.last_event.metadata["objects"]
                found = next((o for o in objects if o["objectId"] == obj_id), None)
                if found:
                    meta_subset = {
                        "objectId": found.get("objectId"),
                        "objectType": found.get("objectType"),
                        "position": found.get("position"),
                        "distance": found.get("distance"),
                        "isOpen": found.get("isOpen"),
                        "isToggled": found.get("isToggled"),
                        "isFilledWithLiquid": found.get("isFilledWithLiquid"),
                        "isDirty": found.get("isDirty"),
                        "isCooked": found.get("isCooked"),
                        "isSliced": found.get("isSliced"),
                        "isBroken": found.get("isBroken"),
                        "isUsedUp": found.get("isUsedUp"),
                        "isPickupable": found.get("pickupable"),
                        "receptacle": found.get("receptacle"),
                        "receptacleObjectIds": found.get("receptacleObjectIds"),
                        "openable": found.get("openable"),
                        "toggleable": found.get("toggleable"),
                        "breakable": found.get("breakable"),
                        "sliceable": found.get("sliceable"),
                        "cookable": found.get("cookable"),
                        "dirtyable": found.get("dirtyable"),
                        "canFillWithLiquid": found.get("canFillWithLiquid"),
                        "canBeUsedUp": found.get("canBeUsedUp"),
                        "parentReceptacles": found.get("parentReceptacles"),
                    }
                    result = {
                        "success": True,
                        "message": json.dumps(meta_subset, indent=2)
                    }
                else:
                    result = {
                        "success": False,
                        "message": f"Object '{obj_id}' not found in current scene."
                    }

            else:
                result = {"success": False, "message": f"Unknown tool: {tool_name}"}

        except Exception as e:
            result = {"success": False, "message": f"Error: {str(e)}"}

        # Update display after any controller action
        try:
            self.agent.display_frame()
        except Exception:
            pass

        return result

    def run(self, goal: str) -> Dict:
        """Run the agent loop to achieve the goal"""
        self.conversation_history = []
        self.navigator.build_reachable_map()
        
        consecutive_failures = 0

        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1}/{self.max_iterations} ---")
            action = self._call_llm(goal)

            if not action:
                print("  [LLM] Failed to parse action.")
                return {"success": False, "message": "Failed to get action from LLM"}

            tool_name = action.get("tool", "")
            arguments = action.get("arguments", {})
            print(f"  [LLM] Tool: {tool_name} | Args: {arguments}")

            # 1. Append assistant action to history
            self.conversation_history.append(
                {
                    "role": "assistant",
                    "content": json.dumps({"tool": tool_name, "arguments": arguments}),
                }
            )

            # 2. Execute tool
            if tool_name == "finish":
                return {"success": True, "message": arguments.get("message", "Task completed")}

            result = self._execute_tool(tool_name, arguments)

            # 3. Get observation and append user feedback
            obs = self.agent.get_context_description()
            feedback = f"Tool: {tool_name}\nResult: {result['message']}\n{obs}"
            self.conversation_history.append({"role": "user", "content": feedback})
            
            # 4. Track consecutive failures to break infinite loops
            if not result.get("success", False):
                consecutive_failures += 1
                if consecutive_failures >= 5:
                    print("  [!] Breaking loop due to 5 consecutive tool failures.")
                    return {"success": False, "message": "Task aborted: Too many consecutive failures."}
            else:
                consecutive_failures = 0

        return {"success": False, "message": "Max iterations reached"}

def create_openai_client(api_key: str = None, model: str = None, base_url: str = None, session_id: str = None, use_langfuse: bool = False):
    """Create an OpenAI client function"""
    # Read from .env defaults if not provided
    if api_key is None:
        api_key = os.getenv("ollama_api_key", "ollama")
    if model is None:
        model = os.getenv("ollama_model", "qwen2.5:14b")
    if base_url is None:
        base_url = os.getenv("ollama_url", "http://ubuntu:11434/v1")

    # Long timeout for Ollama: when multiple models are used (red + blue + judge),
    # Ollama swaps models in/out of VRAM which can take 30-60s. Default OpenAI client
    # timeout is too short and causes "Connection error" during model swaps.
    import httpx
    _ollama_timeout = httpx.Timeout(
        connect=30.0,    # Wait up to 30s for Ollama to accept connection during model swap
        read=120.0,      # Wait up to 2min for model to generate response
        write=30.0,
        pool=30.0,
    )

    if use_langfuse:
        from langfuse.openai import openai
        # Import from the root langfuse module per the new docs
        from langfuse import observe, propagate_attributes

        # Generate a session ID if one isn't provided
        if session_id is None:
            session_id = str(uuid.uuid4())

        # Create client ONCE and reuse across calls
        _traced_client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=_ollama_timeout,
            max_retries=2,
        )

        # 1. Use the @observe decorator directly from langfuse
        @observe(name="ai2thor-agent-step")
        def call_llm_traced(messages: List[Dict], tools: List[Dict]) -> Dict:
            # 2. Use the propagate_attributes context manager to set the session
            with propagate_attributes(session_id=session_id):
                kwargs = dict(model=model, messages=messages)
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"
                return _traced_client.chat.completions.create(**kwargs)

        return call_llm_traced
    else:
        import openai

        # Create client ONCE and reuse across calls (avoids new HTTP connection pool each time)
        _client = openai.OpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=_ollama_timeout,
            max_retries=2,
        )

        def call_llm_standard(messages: List[Dict], tools: List[Dict]) -> Dict:
            # Only pass tools/tool_choice when tools are provided
            # Empty tools list + tool_choice causes API errors
            kwargs = dict(model=model, messages=messages)
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"

            return _client.chat.completions.create(**kwargs)

        return call_llm_standard

