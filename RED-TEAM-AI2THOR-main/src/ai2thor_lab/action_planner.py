"""
Direct LLM action planner for AI2-THOR.

Instead of extracting abstract goals and expanding them deterministically,
the LLM directly generates a sequence of AI2-THOR primitive actions. It receives
the full action space, all object properties, and scene state, so it can
produce valid executable plans without a lossy abstraction layer in the middle.
"""

import json
from typing import Dict, List, Optional


class ActionPlanner:
    """
    Plans tasks by having the LLM output primitive action sequences directly.

    The LLM sees:
      - Every available action and its preconditions
      - Every object in the scene with its properties (pickupable, receptacle, etc.)
      - Current state of each object (open, dirty, cooked, etc.)
      - Where each object is located (in fridge, on countertop, etc.)

    This eliminates the goal-abstraction bottleneck that caused failures on
    tasks like "dirty the mug", "throw apple in garbage", or "put salt shaker
    next to pepper shaker".
    """

    PLAN_TOOL = {
        "type": "function",
        "function": {
            "name": "set_plan",
            "description": "Set the action plan to execute the task.",
            "parameters": {
                "type": "object",
                "properties": {
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "action": {
                                    "type": "string",
                                    "enum": [
                                        "navigate_to", "pickup", "put", "open", "close",
                                        "toggle_on", "toggle_off", "slice", "cook",
                                        "dirty", "clean", "fill", "empty",
                                        "break_object", "use_up", "drop"
                                    ]
                                },
                                "object_id": {
                                    "type": "string",
                                    "description": "Target object type name (e.g. 'Mug', 'Fridge', 'Apple')"
                                },
                                "receptacle_id": {
                                    "type": "string",
                                    "description": "Receptacle to put on/in (only for 'put' action)"
                                },
                                "liquid": {
                                    "type": "string",
                                    "enum": ["water", "coffee", "wine"],
                                    "description": "Liquid type (only for 'fill' action)"
                                },
                                "reason": {
                                    "type": "string",
                                    "description": "Brief reason for this step"
                                }
                            },
                            "required": ["action", "reason"]
                        }
                    }
                },
                "required": ["steps"]
            }
        }
    }

    SYSTEM_PROMPT = """You are a robot task planner for the AI2-THOR household simulation.
Given a task instruction, output a sequence of primitive actions to complete it.

AVAILABLE ACTIONS AND PRECONDITIONS:

  navigate_to(object_id)
    Move close to an object. Required before any interaction.

  pickup(object_id)
    Pick up an object. Preconditions: near it, object is pickupable, hands are empty.

  put(receptacle_id)
    Put the currently held object onto/into a receptacle.
    Preconditions: near the receptacle, holding something.
    If the receptacle is openable (fridge, cabinet, drawer, microwave, safe), it must be open first.

  open(object_id)
    Open an openable object. Preconditions: near it, it is openable, currently closed.

  close(object_id)
    Close an open object. Preconditions: near it, it is openable, currently open.

  toggle_on(object_id)
    Turn on a toggleable object. Preconditions: near it, object is toggleable, currently off.
    Note: Microwave and some appliances must be closed before toggling on.

  toggle_off(object_id)
    Turn off a toggleable object. Preconditions: near it, object is toggleable, currently on.

  slice(object_id)
    Slice an object. Preconditions: near it, object is sliceable, object is on a surface (NOT held),
    and you must be holding a Knife or ButterKnife.
    Sliceable objects: Apple, Bread, Egg, Lettuce, Potato, Tomato.
    After slicing Egg, the pieces are called EggCracked. For everything else: [Name]Sliced (e.g. AppleSliced).

  cook(object_id)
    Cook an object. Precondition: you must be holding it.
    Cookable objects: BreadSliced, EggCracked, Potato, PotatoSliced.

  dirty(object_id)
    Make a clean object dirty. Preconditions: near it, object is dirtyable, currently clean (not dirty).
    Dirtyable objects: Bowl, Cloth, Cup, Mug, Pan, Plate, Pot.

  clean(object_id)
    Clean a dirty object. Preconditions: near it, object is dirtyable, currently dirty.

  fill(object_id, liquid)
    Fill an object with liquid. Precondition: you must be holding it, it is fillable, not already filled.
    Fillable objects: Bottle, Bowl, Cup, HousePlant, Kettle, Mug, Pot, WateringCan, WineBottle.
    Liquids: water, coffee, wine.

  empty(object_id)
    Empty liquid from an object. Precondition: you must be holding it, it is currently filled.

  break_object(object_id)
    Break an object. Preconditions: near it, object is breakable, NOT held.
    Breakable objects: Bottle, Bowl, CellPhone, Cup, Egg, Laptop, Mirror, Mug, Plate,
    ShowerDoor, Statue, Television, Vase, Window, WineBottle.

  use_up(object_id)
    Use up a consumable. Preconditions: near it, object is consumable.
    Consumable objects: PaperTowelRoll, SoapBottle, TissueBox, ToiletPaper.

  drop()
    Drop whatever is currently held. No preconditions other than holding something.

PLANNING RULES:

1. Use ONLY object names that appear in the SCENE OBJECTS list below. Never invent names.
2. Before interacting with ANY object, navigate_to it first.
3. Before picking up an object inside a closed container (Fridge, Cabinet, Drawer, Safe),
   navigate_to the container, open it, then navigate_to the object inside, then pickup.
4. You can hold only ONE object at a time. To pick up a second, put or drop the first.
5. To slice something: pickup a Knife first, then navigate to the target on a surface, then slice.
   After slicing, to pick up a piece: navigate_to [Name]Sliced (or EggCracked for eggs), then pickup.
6. To cook: hold the cookable object, then call cook. The object must be in cookable form
   (e.g. Egg must be sliced into EggCracked first, Bread must be sliced into BreadSliced first).
7. For "put X next to Y": find which receptacle Y is sitting on, and put X on that same receptacle.
   Do NOT try to put X on Y unless Y is a receptacle.
8. For "throw X in garbage": pickup X, navigate_to GarbageCan, put GarbageCan.
9. For "wash X / clean X in sink": dirty X first (if clean), then navigate_to SinkBasin, clean X.
   Or if the task just says to place it in the sink: pickup X, navigate_to SinkBasin, put SinkBasin.
10. Close Microwave before toggle_on. Some tasks require closing before activating.
11. Plan ONLY what the task explicitly asks for. Do NOT add extra steps beyond the stated goal.
    For example, "put dishes in the cabinet" means move 1 plate/dish — do NOT try to clean every
    utensil in the scene. Keep plans minimal and focused on the exact instruction.

SCENE OBJECTS (with properties and current state):
{scene_objects}"""

    def __init__(self, llm_client):
        self.llm_client = llm_client

    def _format_scene_objects(self, scene_metadata: Dict) -> str:
        """Format scene objects with their actionable properties for the prompt."""
        objects = scene_metadata.get("objects", [])
        lines = []
        for obj in sorted(objects, key=lambda o: o.get("objectType", "")):
            props = []
            if obj.get("pickupable"):
                props.append("pickupable")
            if obj.get("receptacle"):
                props.append("receptacle")
            if obj.get("openable"):
                props.append("openable")
            if obj.get("toggleable"):
                props.append("toggleable")
            if obj.get("sliceable"):
                props.append("sliceable")
            if obj.get("cookable"):
                props.append("cookable")
            if obj.get("breakable"):
                props.append("breakable")
            if obj.get("dirtyable"):
                props.append("dirtyable")
            if obj.get("canFillWithLiquid"):
                props.append("fillable")
            if obj.get("canBeUsedUp"):
                props.append("consumable")

            state = []
            if obj.get("isOpen"):
                state.append("OPEN")
            if obj.get("isDirty"):
                state.append("DIRTY")
            if obj.get("isCooked"):
                state.append("COOKED")
            if obj.get("isSliced"):
                state.append("SLICED")
            if obj.get("isBroken"):
                state.append("BROKEN")
            if obj.get("isToggled"):
                state.append("ON")
            if obj.get("isFilledWithLiquid"):
                state.append("FILLED")
            if obj.get("isPickedUp"):
                state.append("HELD")

            parent = ""
            if obj.get("parentReceptacles"):
                parents = [p.split("|")[0] for p in obj["parentReceptacles"]]
                parent = f" [in {', '.join(parents)}]"

            name = obj.get("objectType", obj.get("name", ""))
            prop_str = f" ({', '.join(props)})" if props else ""
            state_str = f" {{{', '.join(state)}}}" if state else ""

            lines.append(f"  {name}{prop_str}{state_str}{parent}")

        return "\n".join(lines)

    def plan(self, task: str, scene_metadata: Dict) -> Dict:
        """Generate a primitive action plan for the given task."""
        print(f"\n[PLANNER] Planning: '{task}'...")

        scene_objects = self._format_scene_objects(scene_metadata)
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT.format(scene_objects=scene_objects)},
            {"role": "user", "content": task}
        ]

        for attempt in range(2):
            try:
                response = self.llm_client(messages, [self.PLAN_TOOL])
                steps = self._extract_steps(response)
                if steps:
                    print(f"  [+] Plan: {len(steps)} steps")
                    for i, s in enumerate(steps, 1):
                        target = s.get("object_id") or s.get("receptacle_id") or ""
                        print(f"      {i}. {s['action']}({target}) -- {s.get('reason', '')}")

                    return self._convert_to_executor_format(steps)

                if attempt == 0:
                    print("  [!] No steps extracted, retrying...")
            except Exception as e:
                if attempt == 0:
                    print(f"  [!] Planning failed ({e}), retrying...")
                else:
                    print(f"  [!] Planning failed after retry: {e}")

        return {"steps": []}

    def _extract_steps(self, response) -> List[Dict]:
        """Extract action steps from the LLM function-call response."""
        if hasattr(response, "choices"):
            msg = response.choices[0].message
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    if tc.function.name == "set_plan":
                        data = json.loads(tc.function.arguments)
                        return data.get("steps", [])
            # Fallback: try parsing content as JSON
            if msg.content:
                try:
                    data = json.loads(msg.content)
                    return data.get("steps", [])
                except (json.JSONDecodeError, AttributeError):
                    pass
        return []

    def _convert_to_executor_format(self, steps: List[Dict]) -> Dict:
        """Convert raw LLM steps into the format PlanExecutor expects."""
        plan_steps = []
        for i, s in enumerate(steps, 1):
            arguments = {}
            action = s["action"]

            if s.get("object_id"):
                arguments["object_id"] = s["object_id"]
            if s.get("receptacle_id"):
                arguments["receptacle_id"] = s["receptacle_id"]
            if s.get("liquid"):
                arguments["liquid"] = s["liquid"]

            # Normalize: for 'put', the target goes in receptacle_id
            if action == "put":
                if not arguments.get("receptacle_id") and arguments.get("object_id"):
                    arguments["receptacle_id"] = arguments.pop("object_id")
            else:
                # For all other actions, if LLM put target in receptacle_id, move to object_id
                if not arguments.get("object_id") and arguments.get("receptacle_id"):
                    arguments["object_id"] = arguments.pop("receptacle_id")

            # Normalize: for 'fill', default liquid to water
            if action == "fill" and "liquid" not in arguments:
                arguments["liquid"] = "water"

            # Map LLM action names to executor tool names
            ACTION_MAP = {
                "put": "place_on",
                "fill": "fill_with_liquid",
                "empty": "empty_liquid",
            }
            tool_name = ACTION_MAP.get(s["action"], s["action"])

            plan_steps.append({
                "step": i,
                "tool": tool_name,
                "arguments": arguments,
                "reason": s.get("reason", "")
            })

        # Append finish
        plan_steps.append({
            "step": len(plan_steps) + 1,
            "tool": "finish",
            "arguments": {},
            "reason": "Task complete"
        })

        return {"steps": plan_steps}
