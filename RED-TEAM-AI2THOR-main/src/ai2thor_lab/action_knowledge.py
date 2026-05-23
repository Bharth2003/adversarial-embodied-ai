"""
Centralized action knowledge base for AI2-THOR interactions.

Distilled from ref-planner/pddl/ domain files and ProblemDefinition object lists.
Provides preconditions, effects, valid object types, and helper functions
so the parser and planner don't need to hardcode per-action validation.
"""

import re


# ---------------------------------------------------------------------------
# ACTION_KNOWLEDGE — one entry per interaction action
#
# Keys:
#   ai2thor_action  : exact Controller.step action string
#   requires_near   : agent must be at an interactable pose
#   object_property : object metadata key that must be True (capability check)
#   blocking_state  : {property: value} — action is blocked when this is True
#   requires_holding: agent must be holding the object first
#   effect          : {property: value} — state change the action produces
#   command_verbs   : regex pattern (word-bounded) to match in natural language
# ---------------------------------------------------------------------------

ACTION_KNOWLEDGE = {
    "pickup": {
        "ai2thor_action": "PickupObject",
        "requires_near": True,
        "object_property": "pickupable",
        "blocking_state": {"isPickedUp": True},
        "requires_holding": False,
        "effect": {"isPickedUp": True},
        "command_verbs": r"\b(pick|grab|take)\b",
    },
    "drop": {
        "ai2thor_action": "DropHandObject",
        "requires_near": False,
        "object_property": None,           # no capability check — just need to hold something
        "blocking_state": None,
        "requires_holding": True,           # PDDL: (holding ?o)
        "effect": {"isPickedUp": False},
        "command_verbs": r"\bdrop\b",
    },
    "open": {
        "ai2thor_action": "OpenObject",
        "requires_near": True,
        "object_property": "openable",
        "blocking_state": {"isOpen": True},
        "requires_holding": False,
        "effect": {"isOpen": True},
        "command_verbs": r"\bopen\b",
    },
    "close": {
        "ai2thor_action": "CloseObject",
        "requires_near": True,
        "object_property": "openable",
        "blocking_state": {"isOpen": False},
        "requires_holding": False,
        "effect": {"isOpen": False},
        "command_verbs": r"\bclose\b",
    },
    "slice": {
        "ai2thor_action": "SliceObject",
        "requires_near": True,
        "object_property": "sliceable",
        "blocking_state": {"isSliced": True},
        "requires_holding": False,
        "effect": {"isSliced": True},
        "command_verbs": r"\b(slice|cut)\b",
    },
    "cook": {
        "ai2thor_action": "CookObject",
        "requires_near": True,
        "object_property": "cookable",
        "blocking_state": {"isCooked": True},
        "requires_holding": False,
        "effect": {"isCooked": True},
        "command_verbs": r"\bcook\b",
    },
    "break": {
        "ai2thor_action": "BreakObject",
        "requires_near": True,
        "object_property": "breakable",
        "blocking_state": {"isBroken": True},
        "requires_holding": False,
        "effect": {"isBroken": True},
        "command_verbs": r"\b(break|smash)\b",
    },
    "dirty": {
        "ai2thor_action": "DirtyObject",
        "requires_near": True,
        "object_property": "dirtyable",
        "blocking_state": {"isDirty": True},
        "requires_holding": False,
        "effect": {"isDirty": True},
        "command_verbs": r"\b(dirty|soil)\b",
    },
    "clean": {
        "ai2thor_action": "CleanObject",
        "requires_near": True,
        "object_property": "dirtyable",
        "blocking_state": {"isDirty": False},
        "requires_holding": False,
        "effect": {"isDirty": False},
        "command_verbs": r"\b(clean|wash)\b",
    },
    "fill": {
        "ai2thor_action": "FillObjectWithLiquid",
        "requires_near": True,
        "object_property": "canFillWithLiquid",
        "blocking_state": {"isFilledWithLiquid": True},
        "requires_holding": False,
        "effect": {"isFilledWithLiquid": True},
        "command_verbs": r"\bfill\b",
    },
    "empty": {
        "ai2thor_action": "EmptyLiquidFromObject",
        "requires_near": True,
        "object_property": "canFillWithLiquid",
        "blocking_state": {"isFilledWithLiquid": False},
        "requires_holding": False,
        "effect": {"isFilledWithLiquid": False},
        "command_verbs": r"\b(empty|pour)\b",
    },
    "toggleon": {
        "ai2thor_action": "ToggleObjectOn",
        "requires_near": True,
        "object_property": "toggleable",
        "blocking_state": {"isToggled": True},
        "requires_holding": False,
        "effect": {"isToggled": True},
        "command_verbs": r"\b(turn on|switch on|toggle on)\b",
    },
    "toggleoff": {
        "ai2thor_action": "ToggleObjectOff",
        "requires_near": True,
        "object_property": "toggleable",
        "blocking_state": {"isToggled": False},
        "requires_holding": False,
        "effect": {"isToggled": False},
        "command_verbs": r"\b(turn off|switch off|toggle off)\b",
    },
    "useup": {
        "ai2thor_action": "UseUpObject",
        "requires_near": True,
        "object_property": "canBeUsedUp",
        "blocking_state": {"isUsedUp": True},
        "requires_holding": False,
        "effect": {"isUsedUp": True},
        "command_verbs": r"\b(use up|useup|consume)\b",
    },
    "put": {
        "ai2thor_action": "PutObject",
        "requires_near": True,
        "object_property": "receptacle",
        "blocking_state": None,
        "requires_holding": True,           # PDDL: (holding ?o)
        "effect": {},
        "command_verbs": r"\b(place|put)\b",
    },
}


# ---------------------------------------------------------------------------
# VALID_OBJECT_TYPES — from ref-planner/core/problem_handler.py
# These serve as a fallback when live scene metadata is unavailable.
# ---------------------------------------------------------------------------

VALID_OBJECT_TYPES = {
    "pickup": [
        "AlarmClock", "AluminumFoil", "Apple", "BaseballBat", "Book", "Boots",
        "BasketBall", "Bottle", "Bowl", "Box", "Bread", "ButterKnife", "Candle",
        "CD", "CellPhone", "PepperShaker", "Cloth", "CreditCard", "Cup",
        "DishSponge", "Dumbbell", "Egg", "Fork", "HandTowel", "Kettle",
        "KeyChain", "Knife", "Ladle", "Laptop", "Lettuce", "Mug", "Newspaper",
        "Pan", "PaperTowel", "PaperTowelRoll", "Pen", "Pencil", "Pillow",
        "Plate", "Plunger", "Pot", "Potato", "RemoteControl", "SaltShaker",
        "ScrubBrush", "SoapBar", "SoapBottle", "Spatula", "Spoon",
        "SprayBottle", "Statue", "TableTopDecor", "TeddyBear", "TennisRacket",
        "TissueBox", "ToiletPaper", "Tomato", "Towel", "Vase", "Watch",
        "WateringCan", "WineBottle",
    ],
    "open": [
        "Blinds", "Book", "Box", "Cabinet", "Drawer", "Fridge", "Kettle",
        "Laptop", "Microwave", "Safe", "ShowerCurtain", "ShowerDoor", "Toilet",
    ],
    "close": [
        "Blinds", "Book", "Box", "Cabinet", "Drawer", "Fridge", "Kettle",
        "Laptop", "Microwave", "Safe", "ShowerCurtain", "ShowerDoor", "Toilet",
    ],
    "break": [
        "Bottle", "Bowl", "CellPhone", "Cup", "Egg", "Laptop", "Television",
        "Mirror", "Mug", "Plate", "ShowerDoor", "Statue", "Vase", "Window",
        "WineBottle",
    ],
    "cook": [
        "BreadSliced", "EggCracked", "Potato", "PotatoSliced",
    ],
    "slice": [
        "Apple", "Bread", "Egg", "Lettuce", "Potato", "Tomato",
    ],
    "toggleon": [
        "Candle", "CellPhone", "CoffeeMachine", "DeskLamp", "Faucet",
        "FloorLamp", "Laptop", "LightSwitch", "Microwave", "ShowerHead",
        "StoveBurner", "StoveKnob", "Television", "Toaster",
    ],
    "toggleoff": [
        "Candle", "CellPhone", "CoffeeMachine", "DeskLamp", "Faucet",
        "FloorLamp", "Laptop", "LightSwitch", "Microwave", "ShowerHead",
        "StoveBurner", "StoveKnob", "Television", "Toaster",
    ],
    "dirty": [
        "Bed", "Bowl", "Cloth", "Cup", "Mirror", "Mug", "Pan", "Plate", "Pot",
    ],
    "clean": [
        "Bed", "Bowl", "Cloth", "Cup", "Mirror", "Mug", "Pan", "Plate", "Pot",
    ],
    "fill": [
        "Bottle", "Bowl", "Cup", "HousePlant", "Kettle", "Mug", "Pot",
        "WateringCan", "WineBottle",
    ],
    "empty": [
        "Bottle", "Bowl", "Cup", "HousePlant", "Kettle", "Mug", "Pot",
        "WateringCan", "WineBottle",
    ],
    "useup": [
        "PaperTowelRoll", "SoapBottle", "TissueBox", "ToiletPaper",
    ],
    "put": [
        "ArmChair", "Bathtub", "BathtubBasin", "Bed", "Bowl", "Box",
        "Cabinet", "CoffeeMachine", "CoffeeTable", "CounterTop", "Desk",
        "DiningTable", "Drawer", "Fridge", "GarbageCan", "HandTowelHolder",
        "LaundryHamper", "Microwave", "Mug", "Ottoman", "Pan", "Plate", "Pot",
        "Safe", "Shelf", "SideTable", "SinkBasin", "Sofa", "Toaster",
        "Toilet", "ToiletPaperHanger", "TowelHolder", "TVStand", "StoveBurner",
    ],
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def validate_action(action_name: str, obj_metadata: dict) -> tuple:
    """Check if an action can be performed on an object given its metadata.

    Args:
        action_name: Key into ACTION_KNOWLEDGE (e.g. "slice", "pickup").
        obj_metadata: Object metadata dict from AI2-THOR (or mock).

    Returns:
        (True, "") if valid, or (False, "reason string") if not.
    """
    knowledge = ACTION_KNOWLEDGE.get(action_name)
    if not knowledge:
        return False, f"Unknown action: {action_name}"

    obj_name = obj_metadata.get("name", obj_metadata.get("objectType", "Object"))

    # Check capability property
    prop = knowledge["object_property"]
    if prop and not obj_metadata.get(prop, False):
        return False, f"{obj_name} cannot be {_past_tense(action_name)}"

    # Check blocking state
    blocking = knowledge["blocking_state"]
    if blocking:
        for key, val in blocking.items():
            if obj_metadata.get(key) == val:
                return False, f"{obj_name} is already {_past_tense(action_name)}"

    return True, ""


def get_action_for_command(cmd_lower: str) -> str | None:
    """Match a natural-language command to an action name.

    Checks multi-word patterns first (e.g. "turn on" before "turn"),
    uses word boundaries to avoid substring collisions.

    Returns:
        Action name key (e.g. "slice") or None if no match.
    """
    # Check multi-word verb patterns first (toggle on/off before single words)
    for action_name in _PRIORITY_ORDER:
        pattern = ACTION_KNOWLEDGE[action_name]["command_verbs"]
        if re.search(pattern, cmd_lower):
            return action_name
    return None


# Order matters: check multi-word patterns before single-word ones
_PRIORITY_ORDER = [
    "toggleon", "toggleoff",     # "turn on/off" before "open"
    "useup",                     # "use up" before generic
    "drop",                      # "drop" before "place" (avoid confusion)
    "put",                       # "place", "put"
    "fill", "empty",
    "slice", "cook", "break",
    "dirty", "clean",
    "open", "close",
    "pickup",
]


def _past_tense(action_name: str) -> str:
    """Simple past-tense for error messages."""
    mapping = {
        "pickup": "picked up",
        "drop": "dropped",
        "open": "open",
        "close": "closed",
        "slice": "sliced",
        "cook": "cooked",
        "break": "broken",
        "dirty": "dirty",
        "clean": "clean",
        "fill": "filled",
        "empty": "empty",
        "toggleon": "toggled on",
        "toggleoff": "toggled off",
        "useup": "used up",
        "put": "placed",
    }
    return mapping.get(action_name, action_name)
