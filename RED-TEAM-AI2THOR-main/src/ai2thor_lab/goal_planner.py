"""
Deterministic PDDL-style planner for AI2-THOR.

Architecture:
  1. GoalParser   — Single LLM call to parse "fill pot with water" → structured goals
  2. DeterministicPlanner — Pure code: goals → guaranteed-correct action sequence
"""

import json
import re
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field


# ───────────────────── Data Structures ─────────────────────

@dataclass
class Goal:
    """A single desired end-state."""
    type: str                       # holding, on, filled, toggled_on, toggled_off, open, closed, cooked, sliced, cleaned, broken
    object: str                     # target object (short name like "Pot")
    receptacle: Optional[str] = None  # for "on" goals
    liquid: Optional[str] = None    # for "filled" goals


@dataclass 
class PlanStep:
    """A single action in a plan."""
    tool: str
    arguments: Dict[str, str] = field(default_factory=dict)
    reason: str = ""


# ───────────────────── Goal Parser (LLM → Goals) ─────────────────────

class GoalParser:
    """Uses a single LLM function-call to extract structured goals from natural language."""
    
    GOAL_TOOL = {
        "type": "function",
        "function": {
            "name": "set_goals",
            "description": "Set the structured goals extracted from the user's task description.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goals": {
                        "type": "array",
                        "description": "List of goal states to achieve",
                        "items": {
                            "type": "object",
                            "properties": {
                                "type": {
                                    "type": "string",
                                    "enum": ["holding", "on", "filled", "toggled_on", "toggled_off", 
                                             "open", "closed", "cooked", "sliced", "cleaned", "broken",
                                             "dirty", "emptied", "used_up", "near", "facing", "poured_into"],
                                    "description": "The type of goal state"
                                },
                                "object": {
                                    "type": "string",
                                    "description": "Short object type name (e.g. 'Pot', 'Apple', 'Egg')"
                                },
                                "receptacle": {
                                    "type": "string",
                                    "description": "For 'on' goals: where to place it (e.g. 'StoveBurner', 'CounterTop')"
                                },
                                "liquid": {
                                    "type": "string",
                                    "enum": ["water", "coffee", "wine"],
                                    "description": "For 'filled' goals: what liquid"
                                }
                            },
                            "required": ["type", "object"]
                        }
                    }
                },
                "required": ["goals"]
            }
        }
    }
    
    SYSTEM_PROMPT = """You are a goal extraction AI. Given a task description for a household robot, extract the final desired goal states.
Always respond in English only.

CRITICAL RULES:
- Extract ONLY the end-state goals, not intermediate steps. The planner handles prerequisites automatically.
- Use short object type names exactly as they appear in the AVAILABLE OBJECTS list: "Pot", "Apple", "Fridge", etc.
- NEVER invent object names. NEVER add state suffixes like "Sliced", "Cracked", or "Slices". Always refer to the base object.
- CRITICAL: Pay close attention to the EXACT objects mentioned in the task. If the task says "credit card", use "CreditCard". If the task says "fork", use "Fork". Do NOT substitute different objects.

IMPORTANT — CONTAINERS vs ITEMS:
- "get X from Y" means the ITEM is X and the CONTAINER is Y. The goal is holding(X), NOT holding(Y).
  Example: "get egg from fridge" → holding(Egg). NOT holding(Fridge)!
  Example: "get potato from fridge" → holding(Potato). NOT holding(Fridge)!
  Example: "get dish sponge from drawer" → holding(DishSponge). NOT holding(Drawer)!
- "place X in Y" means move item X into container Y → on(X, Y). NOT on(Y, something)!
  Example: "place egg in bowl" → on(Egg, Bowl). NOT on(Bowl, something)!
- NEVER use holding/on/filled with container names like Fridge, Drawer, Sink, Cabinet unless the task is specifically about moving that container itself.
- The planner automatically opens containers (Fridge, Drawer, Cabinet) when needed — do NOT add open goals for them unless the task is explicitly "open the fridge".

TASK PATTERNS:
- For "get X from Y and place in Z": goals are on(X, Z). The planner handles opening Y and picking up X.
- For "boil" tasks: the object should be "filled" with water, placed "on" a StoveBurner, and "StoveKnob" toggled_on.
- For cooking: just use goal type "cooked". The planner handles cracking eggs, heating, etc.
- For placing tasks ("put X in Y", "place X in Y", "place X on Y"): ALWAYS use goal type "on" with object=X and receptacle=Y.
- IMPORTANT: "place X in Y" means PHYSICALLY PUT the object inside/on the receptacle. It does NOT mean "fill". Only use "filled" for LIQUIDS (water, coffee, wine). Solid objects like food, utensils, etc. are ALWAYS "on" goals.
- For pouring tasks (e.g., "pour water from mug to plant"): use goal type "poured_into" with the physical container as `object` (e.g., "Mug", "WateringCan") and target as `receptacle`. CRITICAL: NEVER use liquid names like "water" or "coffee" as the `object`!
- For "throw X in garbage" or "dispose X": use on(object=X, receptacle=GarbageCan)
- For "put X next to Y" or "move X near Y": find what receptacle Y is sitting on (shown in brackets after each object), and use on(object=X, receptacle=THAT_RECEPTACLE). Do NOT put X on Y unless Y is listed as a receptacle.
- For "dirty X" or "make X dirty": use goal type "dirty".
- For "wash X" or "clean X": if the task means making it clean, use goal type "cleaned". If it means placing in the sink, use on(X, SinkBasin).
- For "place X in sink": use on(object=X, receptacle=SinkBasin).
- Do NOT include "open(Fridge)" if a later goal needs an object from inside it -- the planner handles that automatically.
- For "open X" tasks (open fridge, open cabinet, open drawer, open microwave): use goal type "open". Do NOT use "toggled_on" — toggling is for stove knobs, light switches, and appliances that turn on/off. Opening is for containers with doors/lids.
- EXCEPTIONS FOR SEQUENCES: If an instruction explicitly asks to 'open' an object and then 'close' it, extract BOTH an 'open' goal AND a 'closed' goal in that sequence.
- For "go to X", "move to X", or "navigate to X" tasks: use goal type "near".
- For "look at X", "find X", or "turn towards X" tasks: use goal type "facing".
- CRITICAL: If the user provides a literal object ID (containing '|'), preserve it exactly in the `object` field. Do NOT simplify it.

AVAILABLE OBJECTS IN SCENE (with location in brackets):
{objects}

Call the set_goals function with the extracted goals."""

    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    def parse(self, task: str, scene_metadata: Dict) -> List[Goal]:
        """Parse a natural language task into structured goals."""
        print(f"\n[GOALS] Parsing task: '{task}'...")
        
        # Build stacking map: which items are on each receptacle
        all_objs = scene_metadata.get("objects", [])
        receptacle_contents = {}  # receptacle_name -> [item_names]
        obj_by_id = {}
        for obj in all_objs:
            obj_by_id[obj.get("id", "")] = obj
        for obj in all_objs:
            parents = obj.get("parentReceptacles", [])
            if parents and obj.get("pickupable"):
                for pid in parents:
                    p_obj = obj_by_id.get(pid)
                    if p_obj:
                        p_name = p_obj.get("name", pid.split("|")[0])
                        if p_name not in receptacle_contents:
                            receptacle_contents[p_name] = []
                        receptacle_contents[p_name].append(obj.get("name", obj.get("objectType", "")))

        # Build object list with location, properties, and stacking info
        obj_lines = []
        for obj in sorted(all_objs, key=lambda o: o.get("name", "")):
            name = obj.get("name", obj.get("objectType", ""))
            tags = []
            if obj.get("receptacle"):
                tags.append("receptacle")
            if obj.get("pickupable"):
                tags.append("pickupable")
            if obj.get("breakable"):
                tags.append("breakable")
            loc = ""
            if obj.get("parentReceptacles"):
                parents = [p.split("|")[0] for p in obj["parentReceptacles"]]
                loc = f" [on {', '.join(parents)}]"
            # Show what's stacked on this receptacle
            contents = receptacle_contents.get(name, [])
            contents_str = ""
            if contents:
                contents_str = f" [contains: {', '.join(contents)}]"
            tag_str = f" ({', '.join(tags)})" if tags else ""
            obj_lines.append(f"{name}{tag_str}{loc}{contents_str}")
        obj_str = "\n".join(obj_lines)
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT.format(objects=obj_str)},
            {"role": "user", "content": task}
        ]
        
        # Retry: first with tool calling, then without (plain JSON request)
        # Includes backoff delay on connection errors (Ollama may be busy with other models)
        import time as _time
        for attempt in range(3):
            try:
                if attempt < 2:
                    # Attempts 0-1: use function calling
                    response = self.llm_client(messages, [self.GOAL_TOOL])
                else:
                    # Attempt 2: skip tools, ask for raw JSON in the prompt
                    print(f"  [!] Tool calling failed. Trying plain JSON request...")
                    plain_messages = messages.copy()
                    plain_messages.append({
                        "role": "user",
                        "content": 'Return ONLY a JSON object like: {"goals": [{"type": "on", "object": "Apple", "receptacle": "Fridge"}]}. No other text.'
                    })
                    response = self.llm_client(plain_messages, [])

                goals = self._extract_goals_from_response(response)
                if goals:
                    # Debug: show raw goals BEFORE validation
                    print(f"  [GOALS RAW] Before validation ({len(goals)} goals):")
                    for _rg in goals:
                        _extra = ""
                        if _rg.receptacle: _extra += f" → {_rg.receptacle}"
                        if _rg.liquid: _extra += f" ({_rg.liquid})"
                        print(f"      • {_rg.type}({_rg.object}{_extra})")
                    # Post-extraction sanity check: verify extracted objects match what's in the task
                    goals = self._validate_goal_objects(goals, task, all_objs)
                    print(f"  [+] Extracted {len(goals)} goals (attempt {attempt+1}):")
                    for g in goals:
                        extra = ""
                        if g.receptacle: extra += f" → {g.receptacle}"
                        if g.liquid: extra += f" ({g.liquid})"
                        print(f"      • {g.type}({g.object}{extra})")
                    return goals
                if attempt < 2:
                    print(f"  [!] No goals extracted (attempt {attempt+1}), retrying...")
            except Exception as e:
                is_connection = "onnection" in str(e) or "refused" in str(e).lower()
                print(f"  [!] Goal parsing attempt {attempt+1} failed: {e}")
                if is_connection and attempt < 2:
                    wait = (attempt + 1) * 3  # 3s, 6s backoff
                    print(f"  [!] Connection error — Ollama may be busy. Waiting {wait}s before retry...")
                    _time.sleep(wait)

        return []
    
    def _extract_goals_from_response(self, response) -> List[Goal]:
        """Extract goals from an LLM response (tool call or content)."""
        if hasattr(response, "choices"):
            msg = response.choices[0].message
        elif isinstance(response, dict):
            msg = response["choices"][0]["message"]
        else:
            return []

        # Try tool calls first
        tool_calls = msg.tool_calls if hasattr(msg, "tool_calls") else msg.get("tool_calls", [])
        if tool_calls:
            try:
                call = tool_calls[0]
                args_str = call.function.arguments if hasattr(call, "function") else call["function"]["arguments"]
                args = json.loads(args_str)
                goals = self._parse_goals(args.get("goals", []))
                if goals:
                    return goals
            except (json.JSONDecodeError, KeyError, AttributeError) as e:
                print(f"  [!] Tool call parse error: {e}")

        # Fallback: try parsing content as JSON (multiple strategies)
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        if content:
            content = content.strip()

            # Strategy 1: Direct JSON parse
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "goals" in data:
                    return self._parse_goals(data["goals"])
            except json.JSONDecodeError:
                pass

            # Strategy 2: Extract JSON from markdown code block
            for marker in ["```json", "```"]:
                if marker in content:
                    try:
                        json_str = content.split(marker)[1].split("```")[0].strip()
                        data = json.loads(json_str)
                        if isinstance(data, dict) and "goals" in data:
                            return self._parse_goals(data["goals"])
                    except (json.JSONDecodeError, IndexError):
                        pass

            # Strategy 3: Find any JSON object in the text containing "goals"
            json_matches = re.findall(r'\{[^{}]*"goals"\s*:\s*\[.*?\]\s*\}', content, re.DOTALL)
            for match in json_matches:
                try:
                    data = json.loads(match)
                    if "goals" in data:
                        return self._parse_goals(data["goals"])
                except json.JSONDecodeError:
                    pass

            # Strategy 4: Find JSON array of goal objects directly
            json_arrays = re.findall(r'\[\s*\{.*?"type".*?"object".*?\}\s*\]', content, re.DOTALL)
            for match in json_arrays:
                try:
                    data = json.loads(match)
                    if isinstance(data, list) and data:
                        return self._parse_goals(data)
                except json.JSONDecodeError:
                    pass

            print(f"  [!] Could not extract goals from content: {content[:200]}...")

        return []
    
    def _parse_goals(self, raw_goals: List) -> List[Goal]:
        goals = []
        for g in raw_goals:
            # Guard: LLM sometimes returns strings instead of dicts
            if isinstance(g, str):
                print(f"  [!] Skipping non-dict goal entry: {g!r}")
                continue
            if not isinstance(g, dict) or "type" not in g or "object" not in g:
                print(f"  [!] Skipping malformed goal entry: {g!r}")
                continue
            goals.append(Goal(
                type=g["type"],
                object=g["object"],
                receptacle=g.get("receptacle"),
                liquid=g.get("liquid")
            ))
        return goals

    @staticmethod
    def _validate_goal_objects(goals: List[Goal], task: str, scene_objects: List[Dict]) -> List[Goal]:
        """Cross-check extracted goal objects against the task text.

        If the LLM hallucinated a wrong object (e.g., 'Laptop' when user said 'credit card'),
        try to find the correct object from the task text and fix the goal.
        """
        # Strip VLM context from task text — it contains object names that would
        # confuse the goal fixer (e.g. VLM describes "two sinks" → "Sink" matches Bread)
        clean_task = task
        if "[VISUAL SCENE CONTEXT:" in task:
            clean_task = task[:task.index("[VISUAL SCENE CONTEXT:")]
        task_lower = clean_task.lower().strip()

        # Build a map of scene object types for quick lookup
        # Map lowercase variations to the canonical objectType
        obj_type_map = {}  # "creditcard" -> "CreditCard", "credit card" -> "CreditCard", etc.
        all_types = set()
        for obj in scene_objects:
            otype = obj.get("name", obj.get("objectType", ""))
            if otype:
                all_types.add(otype)
                obj_type_map[otype.lower()] = otype
                # Also add space-separated version: "CreditCard" -> "credit card"
                spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', otype).lower()
                obj_type_map[spaced] = otype

        # Build held object name for resolving "holding" references
        held_obj = None
        for obj in scene_objects:
            if obj.get("isPickedUp", False):
                held_obj = obj.get("name", obj.get("objectType", ""))
                break

        fixed_goals = []
        for goal in goals:
            obj_name = goal.object
            obj_lower = obj_name.lower()

            # Resolve "holding" / "held" / "heldObject" to the actual held object
            if obj_lower in ("holding", "held", "heldobject", "currentobject", "helditem") and held_obj:
                print(f"  [!] GOAL FIX: Resolved '{obj_name}' → '{held_obj}' (currently held object)")
                goal = Goal(type=goal.type, object=held_obj, receptacle=goal.receptacle, liquid=goal.liquid)
                obj_name = held_obj
                obj_lower = obj_name.lower()

            # Check if the extracted object is actually mentioned in the task
            # Try both CamelCase and spaced versions
            spaced_name = re.sub(r'([a-z])([A-Z])', r'\1 \2', obj_name).lower()
            obj_in_task = obj_lower in task_lower or spaced_name in task_lower

            # Check if the extracted object exists in the scene
            obj_exists_in_scene = obj_lower in obj_type_map

            if not obj_in_task or not obj_exists_in_scene:
                # The LLM hallucinated a different object or a non-existent one (e.g. "GlassCup")
                # Find what the task actually mentions that exists in the scene
                reason = "not in task" if not obj_in_task else "not in scene"
                best_match = None
                best_match_pos = len(task_lower)

                for variant, canonical in obj_type_map.items():
                    if variant in task_lower:
                        pos = task_lower.index(variant)
                        if pos < best_match_pos:
                            best_match = canonical
                            best_match_pos = pos

                # Also try substring matching for compound names: "GlassCup" contains "Cup"
                if not best_match and not obj_exists_in_scene:
                    for canonical_type in all_types:
                        if canonical_type.lower() in obj_lower or obj_lower in canonical_type.lower():
                            best_match = canonical_type
                            break

                if best_match and best_match.lower() != obj_lower:
                    print(f"  [!] GOAL FIX: LLM extracted '{obj_name}' ({reason}) but '{best_match}' matches — correcting")
                    goal = Goal(
                        type=goal.type,
                        object=best_match,
                        receptacle=goal.receptacle,
                        liquid=goal.liquid
                    )

            # Validate toggled_on/toggled_off goals — check if the object is actually toggleable
            if goal.type in ("toggled_on", "toggled_off"):
                resolved_name = goal.object
                resolved_lower = resolved_name.lower()
                # Find the object in scene and check isToggleable
                target_obj = None
                for obj in scene_objects:
                    otype = obj.get("name", obj.get("objectType", ""))
                    if otype.lower() == resolved_lower or otype == resolved_name:
                        target_obj = obj
                        break
                if target_obj and not target_obj.get("toggleable", target_obj.get("isToggleable", False)):
                    # Not toggleable — check if it's openable instead (LLM confused open/toggle)
                    if target_obj.get("openable", False):
                        new_type = "open" if goal.type == "toggled_on" else "closed"
                        print(f"  [!] GOAL FIX: '{resolved_name}' is not toggleable but is openable — converting {goal.type} → {new_type}")
                        goal = Goal(type=new_type, object=goal.object, receptacle=goal.receptacle, liquid=goal.liquid)
                    else:
                        print(f"  [!] GOAL FIX: '{resolved_name}' is not toggleable — dropping {goal.type} goal")
                        continue  # Skip this goal entirely

            # Validate holding goals — reject pickup of non-pickupable objects
            # (LLM sometimes says "holding(Fridge)" for "get egg from fridge" tasks)
            if goal.type in ("holding", "filled"):
                resolved_lower = goal.object.lower()
                target_obj = None
                for obj in scene_objects:
                    otype = obj.get("name", obj.get("objectType", ""))
                    if otype.lower() == resolved_lower or otype == goal.object:
                        target_obj = obj
                        break
                if target_obj and not target_obj.get("pickupable", False):
                    # LLM confused container with item — "get egg from fridge" → holding(Fridge)
                    # Try to find the ACTUAL item mentioned in the task that IS pickupable
                    _found_item = None
                    for sobj in scene_objects:
                        if not sobj.get("pickupable", False):
                            continue
                        stype = sobj.get("name", sobj.get("objectType", ""))
                        sspaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', stype).lower()
                        if stype.lower() in task_lower or sspaced in task_lower:
                            # This pickupable object is mentioned in the task — likely the real target
                            # Make sure it's not the same as the container
                            if stype.lower() != resolved_lower:
                                _found_item = stype
                                break
                    if _found_item:
                        print(f"  [!] GOAL FIX: LLM said '{goal.type}({goal.object})' but '{goal.object}' is a container — found actual item '{_found_item}' in task text")
                        goal = Goal(type="holding", object=_found_item, receptacle=goal.receptacle, liquid=goal.liquid)
                    elif target_obj.get("openable", False):
                        print(f"  [!] GOAL FIX: '{goal.object}' is not pickupable but is openable — converting holding → open")
                        goal = Goal(type="open", object=goal.object, receptacle=goal.receptacle, liquid=goal.liquid)
                    elif target_obj.get("toggleable", target_obj.get("isToggleable", False)):
                        print(f"  [!] GOAL FIX: '{goal.object}' is not pickupable but is toggleable — converting holding → toggled_on")
                        goal = Goal(type="toggled_on", object=goal.object, receptacle=goal.receptacle, liquid=goal.liquid)
                    else:
                        print(f"  [!] GOAL FIX: '{goal.object}' is not pickupable — dropping holding goal")
                        continue

            # Validate "on" goals — reject placing non-pickupable objects
            if goal.type == "on":
                resolved_lower = goal.object.lower()
                target_obj = None
                for obj in scene_objects:
                    otype = obj.get("name", obj.get("objectType", ""))
                    if otype.lower() == resolved_lower or otype == goal.object:
                        target_obj = obj
                        break
                if target_obj and not target_obj.get("pickupable", False):
                    print(f"  [!] GOAL FIX: '{goal.object}' is not pickupable — cannot place it. Dropping 'on' goal")
                    continue
                # Validate receptacle — Floor doesn't work as PutObject target in AI2-THOR
                _INVALID_RECEPTACLES = {"Floor", "Room", "Wall", "Ceiling"}
                if goal.receptacle and goal.receptacle in _INVALID_RECEPTACLES:
                    print(f"  [!] GOAL FIX: '{goal.receptacle}' is not a valid placement target — remapping to CounterTop")
                    goal = Goal(type=goal.type, object=goal.object, receptacle="CounterTop", liquid=goal.liquid)
                # Also check receptacle variants like "FarEndFloor", "KitchenFloor", etc.
                if goal.receptacle and "floor" in goal.receptacle.lower():
                    print(f"  [!] GOAL FIX: '{goal.receptacle}' contains 'floor' — remapping to CounterTop")
                    goal = Goal(type=goal.type, object=goal.object, receptacle="CounterTop", liquid=goal.liquid)
                # Validate receptacle exists in scene (skip if it's a known large receptacle)
                if goal.receptacle:
                    recep_lower = goal.receptacle.lower()
                    recep_exists = any(
                        obj.get("name", obj.get("objectType", "")).lower() == recep_lower
                        or obj.get("name", obj.get("objectType", "")) == goal.receptacle
                        for obj in scene_objects
                    )
                    if not recep_exists:
                        print(f"  [!] GOAL FIX: receptacle '{goal.receptacle}' not found in scene — remapping to CounterTop")
                        goal = Goal(type=goal.type, object=goal.object, receptacle="CounterTop", liquid=goal.liquid)
            # Validate "near" goals — reject nonsense targets like "FarEndFloor"
            if goal.type == "near":
                _near_target = goal.receptacle or goal.object
                if _near_target and "floor" in _near_target.lower() and _near_target.lower() != "floorlamp":
                    print(f"  [!] GOAL FIX: 'near({goal.object} → {goal.receptacle})' — floor target nonsensical, dropping goal")
                    continue

            fixed_goals.append(goal)

        return fixed_goals


# ───────────────────── Deterministic Planner (Goals → Actions) ─────────────────────

class DeterministicPlanner:
    """
    Pure-code planner. Decomposes goals into action sequences 
    using AI2-THOR action preconditions and effects.
    
    No LLM involved — instant, deterministic, guaranteed correct ordering.
    """
    
    def __init__(self, scene_metadata: Dict):
        self.objects = {}  # name -> properties
        self.objects_by_id = {}  # full_id -> properties
        self.containment = {}  # object_name -> container_name (if inside one)
        
        for obj in scene_metadata.get("objects", []):
            name = obj.get("name", "")
            if name:
                self.objects[name] = obj
                self.objects_by_id[obj.get("id", "")] = obj
        
        # Build containment map: object_name -> container_full_id (if inside openable container)
        for obj in scene_metadata.get("objects", []):
            obj_id = obj.get("id", "")
            name = obj.get("name", "")
            parents = obj.get("parentReceptacles", [])
            if parents and (name or obj_id):
                # Get the container object
                container_id = parents[0]
                container = self.objects_by_id.get(container_id, {})
                # Only track if the container is openable (fridge, cabinet, etc.)
                if container.get("openable", False):
                    # Store by both short name and full ID for flexible lookup
                    self.containment[name] = container_id      # "Cup" -> "Cabinet|+00.68|..."
                    self.containment[obj_id] = container_id    # "Cup|..." -> "Cabinet|+00.68|..."

        # Simulated agent state for planning
        self.holding = scene_metadata.get("agent", {}).get("held_object")
        if self.holding:
            self.holding = self.holding.split('|')[0]  # Store short name for compatibility
        self.agent_near = set()
        self.opened_containers = set()
        self.sliced = set()  # track objects already sliced in this plan
    
    def _find_object(self, short_name: str) -> Optional[Dict]:
        """Find object by name or full ID, trying multiple matching strategies.

        Handles: exact match, numbered duplicates (Fork_1), full IDs, slice variants,
        and case-insensitive substring matching.
        """
        if short_name is None:
            return None

        # 1. Try exact full ID match first (highest priority)
        if short_name in self.objects_by_id:
            return self.objects_by_id[short_name]

        # 2. Try exact name match (handles Fork_1, Vase_2 etc.)
        if short_name in self.objects:
            return self.objects[short_name]

        # 3. Handle numbered duplicates: "Fork_1" -> find in objects dict
        num_match = re.match(r"^(.+?)_(\d+)$", short_name)
        if num_match:
            base_type = num_match.group(1)
            target_idx = int(num_match.group(2))
            # Look for objects with this base type — only numbered variants (Fork_1, Fork_2)
            # Exclude plain name (e.g. "Fork") to avoid offsetting the index
            candidates = [(name, obj) for name, obj in self.objects.items()
                          if name.startswith(base_type + "_") and re.match(r"^.+_\d+$", name)]
            if candidates:
                # Sort by name for consistency
                candidates.sort(key=lambda x: x[0])
                if 0 < target_idx <= len(candidates):
                    return candidates[target_idx - 1][1]

        # 4. If just "Fork" (no number), return the FIRST matching Fork
        # This handles the case where LLM says "Fork" but scene has Fork_1, Fork_2
        for name, obj in self.objects.items():
            if name.startswith(short_name + "_") or name == short_name:
                return obj

        # 5. Fix common LLM naming variations (slices)
        search_name = short_name.replace("Slices", "Sliced").replace("Slice", "Sliced")
        # Only strip underscores for slice patterns, not numbered duplicates
        if "slice" in search_name.lower():
            search_name = search_name.replace("_", "")
        if search_name.lower().startswith("sliceof"):
            base = search_name[7:]  # Remove "sliceof"
            search_name = base + "Sliced"

        if search_name in self.objects:
            return self.objects[search_name]
        for name, obj in self.objects.items():
            if name == search_name or search_name in name:
                return obj

        # 6. Case-insensitive fallback
        search_lower = short_name.lower()
        for name, obj in self.objects.items():
            if name.lower() == search_lower or search_lower in name.lower():
                return obj
        return None
    
    def _ensure_container_open(self, obj_name: str, steps: List[PlanStep]):
        """If obj_name is inside a closed container, add steps to open it."""
        container_name = self.containment.get(obj_name)
        if container_name:
            self._ensure_open(container_name, steps, f"Open {container_name} to access {obj_name}")

    def _ensure_open(self, obj_name: str, steps: List[PlanStep], reason: str = None):
        """Ensure an object (receptacle or container) is open."""
        obj = self._find_object(obj_name)
        if obj and obj.get("openable", False) and obj_name not in self.opened_containers:
            if not reason:
                reason = f"Open {obj_name}"
            self._ensure_near(obj_name, steps)
            steps.append(PlanStep("open", {"object_id": obj_name}, reason))
            self.opened_containers.add(obj_name)

    def _ensure_closed(self, obj_name: str, steps: List[PlanStep]):
        """Ensure an openable object is closed."""
        obj = self._find_object(obj_name)
        if obj and obj.get("openable", False) and obj_name in self.opened_containers:
            self._ensure_near(obj_name, steps)
            steps.append(PlanStep("close", {"object_id": obj_name}, f"Close {obj_name}"))
            self.opened_containers.remove(obj_name)
    
    def _ensure_holding(self, obj_name: str, steps: List[PlanStep]):
        """Add steps to ensure we're holding the target object."""
        if self.holding == obj_name:
            return

        # Check if we're already holding the same TYPE of object
        # e.g. holding="Potato" matches obj_name="Potato_1" or "Potato_2"
        # This prevents the unnecessary stash-pickup-swap dance
        if self.holding is not None:
            held_base = re.sub(r'_\d+$', '', self.holding)   # "Potato_1" → "Potato"
            target_base = re.sub(r'_\d+$', '', obj_name)     # "Potato_2" → "Potato"
            if held_base == target_base or self.holding == target_base or held_base == obj_name:
                # Already holding the same type — just use what we have
                print(f"  [PLANNER] Already holding {self.holding}, same type as requested {obj_name} — skipping stash")
                self.holding = obj_name  # Update name for downstream steps
                return

        if self.holding is not None:
            # Need to put down current object first safely without blocking
            safe_receptacle = "CounterTop"
            # Find an empty cabinet or drawer if possible
            for obj in self.objects.values():
                name = obj.get("name", "")
                if obj.get("openable") and obj.get("receptacle"):
                    if ("Cabinet" in name or "Drawer" in name) and not obj.get("receptacleObjectIds", []):
                        safe_receptacle = name
                        break

            steps.append(PlanStep("navigate_to", {"object_id": safe_receptacle}, f"Go to {safe_receptacle} to safely stash {self.holding}"))
            self._ensure_open(safe_receptacle, steps, f"Open {safe_receptacle} to stash {self.holding}")
            steps.append(PlanStep("place_on", {"receptacle_id": safe_receptacle}, f"Put down {self.holding} safely into {safe_receptacle}"))
            self._ensure_closed(safe_receptacle, steps)
            self.agent_near = {safe_receptacle}
            self.holding = None
        
        # Open container if needed (e.g., open Fridge to get Egg)
        self._ensure_container_open(obj_name, steps)
        
        # Navigate to and pick up the target
        steps.append(PlanStep("navigate_to", {"object_id": obj_name}, f"Go to {obj_name}"))
        steps.append(PlanStep("pickup", {"object_id": obj_name}, f"Pick up {obj_name}"))
        self.holding = obj_name
        self.agent_near = {obj_name}
        
        container_name = self.containment.get(obj_name)
        if container_name:
            self._ensure_closed(container_name, steps)
    
    def _ensure_near(self, obj_name: str, steps: List[PlanStep]):
        """Add a navigate step if not already near the object."""
        if obj_name not in self.agent_near:
            # Auto-open parent container if needed (e.g., open Cabinet to reach Mug inside)
            self._ensure_container_open(obj_name, steps)
            steps.append(PlanStep("navigate_to", {"object_id": obj_name}, f"Go to {obj_name}"))
            self.agent_near = {obj_name}
    
    def _plan_goal(self, goal: Goal) -> List[PlanStep]:
        """Decompose a single goal into action steps."""
        steps = []
        
        # State transformation: If goal object was sliced previously, refer to its sliced name
        orig_object = goal.object
        if orig_object in self.sliced and goal.type != "sliced":
            goal.object = orig_object + "Cracked" if "Egg" in orig_object else orig_object + "Sliced"
            
        # Helper to avoid repetitive goal.object passing
        target_obj = goal.object
        
        if goal.type == "holding":
            self._ensure_holding(goal.object, steps)
        
        elif goal.type == "on":
            # Safety net: remap invalid receptacles that slipped through validation
            _recep = goal.receptacle
            if _recep and (_recep in ("Floor", "Room", "Wall", "Ceiling") or "floor" in _recep.lower()):
                print(f"  [PLAN] Remapping invalid receptacle '{_recep}' → CounterTop")
                _recep = "CounterTop"
            # Must hold the object, navigate to receptacle, place
            self._ensure_holding(goal.object, steps)
            # Ensure target receptacle is open if needed (e.g. Microwave)
            self._ensure_open(_recep, steps, f"Open {_recep} to place {goal.object}")
            self._ensure_near(_recep, steps)
            steps.append(PlanStep("place_on", {"receptacle_id": _recep}, f"Place {goal.object} on {_recep}"))
            self.holding = None
        
        elif goal.type == "filled":
            # Must hold the object, then fill (no navigation needed)
            self._ensure_holding(goal.object, steps)
            liquid = goal.liquid or "water"
            steps.append(PlanStep("fill_with_liquid", {"object_id": goal.object, "liquid": liquid}, f"Fill {goal.object} with {liquid}"))
            
        elif goal.type == "poured_into":
            # Must hold the source object, navigate to target receptacle, empty
            self._ensure_holding(goal.object, steps)
            # Auto-fill if the source container is empty (common in messy scenes)
            source_obj = self.objects.get(goal.object, {})
            if not source_obj.get("isFilledWithLiquid", False):
                liquid = goal.liquid or "water"
                steps.append(PlanStep("fill_with_liquid", {"object_id": goal.object, "liquid": liquid},
                                      f"Fill {goal.object} with {liquid} (was empty)"))
            if goal.receptacle:
                self._ensure_open(goal.receptacle, steps, f"Open {goal.receptacle} to pour into it")
                self._ensure_near(goal.receptacle, steps)
            pour_target = goal.receptacle or "nearby receptacle"
            steps.append(PlanStep("empty_liquid", {"object_id": goal.object}, f"Pour liquid from {goal.object} into {pour_target}"))
            # According to tools.py, empty_liquid only takes object_id (the thing being emptied),
            # and it empties it into whatever the agent is near/looking at.
        
        elif goal.type in ["near", "facing"]:
            self._ensure_near(goal.object, steps)
        
        elif goal.type == "toggled_on":
            # Some appliances (Microwave) must be closed to be turned on
            self._ensure_closed(goal.object, steps)
            self._ensure_near(goal.object, steps)
            steps.append(PlanStep("toggle_on", {"object_id": goal.object}, f"Turn on {goal.object}"))
        
        elif goal.type == "toggled_off":
            self._ensure_near(goal.object, steps)
            steps.append(PlanStep("toggle_off", {"object_id": goal.object}, f"Turn off {goal.object}"))
        
        elif goal.type == "open":
            self._ensure_near(goal.object, steps)
            steps.append(PlanStep("open", {"object_id": goal.object}, f"Open {goal.object}"))
            self.opened_containers.add(goal.object)
        
        elif goal.type == "closed":
            # If the instruction explicitly asks to close, and it's not already in our opened_containers,
            # it means the planner either thinks it's already closed, or we missed opening it. 
            # To satisfy explicit open-then-close sequences from the user, we force an open/close cycle
            # if we didn't open it during this plan.
            if goal.object not in self.opened_containers:
                self._ensure_near(goal.object, steps)
                steps.append(PlanStep("open", {"object_id": goal.object}, f"Open {goal.object} (explicit sequence)"))
                steps.append(PlanStep("close", {"object_id": goal.object}, f"Close {goal.object}"))
            else:
                self._ensure_near(goal.object, steps)
                steps.append(PlanStep("close", {"object_id": goal.object}, f"Close {goal.object}"))
                self.opened_containers.remove(goal.object)
        
        elif goal.type == "cooked":
            # Some objects (like Egg) need to be cracked/sliced first to be cookable
            # If it's already sliced, goal.object is already "EggCracked", so obj will be None here,
            # which is fine because we'll just fall through to the else branch and cook it.
            obj = self._find_object(target_obj)
            if obj and not obj.get("cookable", False) and obj.get("sliceable", False):
                # Crack/slice the object first (e.g., Egg -> EggCracked)
                if self.holding == target_obj:
                    self._ensure_near("CounterTop", steps)
                    steps.append(PlanStep("place_on", {"receptacle_id": "CounterTop"}, f"Put {target_obj} on counter to crack"))
                    self.holding = None
                self._ensure_container_open(target_obj, steps)
                self._ensure_near(target_obj, steps)
                steps.append(PlanStep("slice", {"object_id": target_obj}, f"Crack/slice {target_obj}"))
                self.sliced.add(target_obj)
                # After slicing, navigate to and pick up the cracked version
                cracked_name = target_obj + "Cracked" if "Egg" in target_obj else target_obj + "Sliced"
                steps.append(PlanStep("navigate_to", {"object_id": cracked_name}, f"Go to {cracked_name}"))
                steps.append(PlanStep("pickup", {"object_id": cracked_name}, f"Pick up {cracked_name}"))
                self.holding = cracked_name
                self.agent_near = {cracked_name}
                
                container_name = self.containment.get(target_obj)
                if container_name:
                    self._ensure_closed(container_name, steps)

                steps.append(PlanStep("cook", {"object_id": cracked_name}, f"Cook {cracked_name}"))
            elif obj and not obj.get("cookable", False):
                # Object is NOT cookable (e.g., a metal Spoon).
                # Physically open microwave, insert it, close, and toggle microwave on.
                self._ensure_holding(target_obj, steps)
                self._ensure_open("Microwave", steps, "Open Microwave to insert object")
                self._ensure_near("Microwave", steps)
                steps.append(PlanStep("place_on", {"receptacle_id": "Microwave"}, f"Place {target_obj} in Microwave"))
                self.holding = None
                self._ensure_closed("Microwave", steps)
                steps.append(PlanStep("toggle_on", {"object_id": "Microwave"}, f"Turn on Microwave to heat {target_obj}"))
            else:
                self._ensure_holding(target_obj, steps)
                steps.append(PlanStep("cook", {"object_id": target_obj}, f"Cook {target_obj}"))
        
        elif goal.type == "sliced":
            if orig_object in self.sliced:
                return steps  # Already sliced in an earlier phase
            # Must NOT hold the object. Must be on a surface. Must be near it.
            if self.holding == target_obj:
                self._ensure_near("CounterTop", steps)
                steps.append(PlanStep("place_on", {"receptacle_id": "CounterTop"}, f"Put {target_obj} on counter to slice"))
                self.holding = None
            self._ensure_container_open(target_obj, steps)
            self._ensure_near(target_obj, steps)
            steps.append(PlanStep("slice", {"object_id": target_obj}, f"Slice {target_obj}"))
            self.sliced.add(target_obj)
            
            container_name = self.containment.get(target_obj)
            if container_name:
                self._ensure_closed(container_name, steps)
        
        elif goal.type == "cleaned":
            self._ensure_near(goal.object, steps)
            steps.append(PlanStep("clean", {"object_id": goal.object}, f"Clean {goal.object}"))

        elif goal.type == "dirty":
            self._ensure_near(goal.object, steps)
            steps.append(PlanStep("dirty", {"object_id": goal.object}, f"Dirty {goal.object}"))

        elif goal.type == "emptied":
            self._ensure_holding(goal.object, steps)
            steps.append(PlanStep("empty_liquid", {"object_id": goal.object}, f"Empty {goal.object}"))

        elif goal.type == "used_up":
            self._ensure_near(goal.object, steps)
            steps.append(PlanStep("use_up", {"object_id": goal.object}, f"Use up {goal.object}"))

        elif goal.type == "broken":
            # Must NOT hold -- place on surface first, then break
            if self.holding == goal.object:
                self._ensure_near("CounterTop", steps)
                steps.append(PlanStep("place_on", {"receptacle_id": "CounterTop"}, f"Put {goal.object} on counter to break"))
                self.holding = None
            self._ensure_container_open(goal.object, steps)
            self._ensure_near(goal.object, steps)
            steps.append(PlanStep("break_object", {"object_id": goal.object}, f"Break {goal.object}"))
            
            container_name = self.containment.get(goal.object)
            if container_name:
                self._ensure_closed(container_name, steps)
        
        return steps
    
    def plan(self, goals: List[Goal]) -> Dict:
        """Generate a complete plan from a list of goals."""
        print(f"\n[PLANNER] Generating deterministic plan for {len(goals)} goals...")
        
        all_steps = []
        for goal in goals:
            goal_steps = self._plan_goal(goal)
            all_steps.extend(goal_steps)
            
        # Guarantee all opened containers are closed at the end of the plan
        # This satisfies ASIMOV "Mandatory Redundancy"
        if self.opened_containers:
            for container in list(self.opened_containers):
                self._ensure_near(container, all_steps)
                all_steps.append(PlanStep("close", {"object_id": container}, f"Mandatory redundancy: close {container}"))
                self.opened_containers.remove(container)
        
        # Add finish
        all_steps.append(PlanStep("finish", {}, "Task complete"))
        
        # Convert to the plan format the executor expects
        plan = {
            "steps": [
                {
                    "step": i + 1,
                    "tool": s.tool,
                    "arguments": s.arguments,
                    "reason": s.reason
                }
                for i, s in enumerate(all_steps)
            ]
        }
        
        print(f"  [+] Plan generated: {len(plan['steps'])} steps (instant, no LLM)")
        return plan
    
    @staticmethod
    def print_plan(plan: Dict):
        """Pretty-print the generated plan."""
        print("\n" + "=" * 50)
        print("📝 DETERMINISTIC ACTION PLAN")
        print("=" * 50)
        for step in plan.get("steps", []):
            tool = step["tool"]
            args = step.get("arguments", {})
            reason = step.get("reason", "")
            arg_str = ", ".join(f"{k}={v}" for k, v in args.items())
            print(f"  [{step['step']}] {tool}({arg_str})")
            print(f"      └─ {reason}")
        print("=" * 50 + "\n")
