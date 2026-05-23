TOOLS = [
    {
        "name": "navigate_to",
        "description": "Move the agent to a specific object. The agent will pathfind to the object's location.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "The EXACT object ID from the provided scene list (e.g., 'Sink|-01.90|+00.97|-01.50|SinkBasin', 'Apple|-01.20|+00.90|-01.30')",
                }
            },
            "required": ["object_id"],
        },
    },
    {
        "name": "pickup",
        "description": "Pick up an object. Agent must be close to it and it must be pickupable.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "The EXACT object ID from the provided list to pick up",
                }
            },
            "required": ["object_id"],
        },
    },
    {
        "name": "drop",
        "description": "Drop the currently held object.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "place_on",
        "description": "Place the held object onto a receptacle.",
        "parameters": {
            "type": "object",
            "properties": {
                "receptacle_id": {
                    "type": "string",
                    "description": "The EXACT receptacle ID from the provided list to place the object on",
                }
            },
            "required": ["receptacle_id"],
        },
    },
    {
        "name": "open",
        "description": "Open an object (e.g., fridge, cabinet, microwave).",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {"type": "string", "description": "The object ID to open"}
            },
            "required": ["object_id"],
        },
    },
    {
        "name": "close",
        "description": "Close an object (e.g., fridge, cabinet, microwave).",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {"type": "string", "description": "The object ID to close"}
            },
            "required": ["object_id"],
        },
    },
    {
        "name": "toggle_on",
        "description": "Turn on a toggleable object (faucet, stoveknob, microwave, light).",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "The object ID to turn on",
                }
            },
            "required": ["object_id"],
        },
    },
    {
        "name": "toggle_off",
        "description": "Turn off a toggleable object (faucet, stoveknob, microwave, light).",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "The object ID to turn off",
                }
            },
            "required": ["object_id"],
        },
    },
    {
        "name": "slice",
        "description": "Slice a sliceable object (apple, potato, bread).",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {"type": "string", "description": "The object ID to slice"}
            },
            "required": ["object_id"],
        },
    },
    {
        "name": "cook",
        "description": "Cook a cookable object (potato, egg).",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {"type": "string", "description": "The object ID to cook"}
            },
            "required": ["object_id"],
        },
    },
    {
        "name": "fill_with_liquid",
        "description": "Fill an object with a liquid.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {"type": "string", "description": "The object ID to fill"},
                "liquid": {
                    "type": "string",
                    "enum": ["water", "coffee", "wine"],
                    "description": "Type of liquid to fill with",
                },
            },
            "required": ["object_id", "liquid"],
        },
    },
    {
        "name": "empty_liquid",
        "description": "Empty liquid from an object.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {"type": "string", "description": "The object ID to empty"}
            },
            "required": ["object_id"],
        },
    },
    {
        "name": "clean",
        "description": "Clean a dirty object.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {"type": "string", "description": "The object ID to clean"}
            },
            "required": ["object_id"],
        },
    },
    {
        "name": "dirty",
        "description": "Dirty a clean object.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {"type": "string", "description": "The object ID to dirty"}
            },
            "required": ["object_id"],
        },
    },
    {
        "name": "break_object",
        "description": "Break a breakable object (egg).",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {"type": "string", "description": "The object ID to break"}
            },
            "required": ["object_id"],
        },
    },
    {
        "name": "use_up",
        "description": "Use up a consumable object (toilet paper).",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "The object ID to use up",
                }
            },
            "required": ["object_id"],
        },
    },
    {
        "name": "look_up",
        "description": "Look up (rotate camera upward).",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "look_down",
        "description": "Look down (rotate camera downward).",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "look_straight",
        "description": "Look straight forward (reset camera horizon).",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "rotate_left",
        "description": "Rotate agent left by specified degrees.",
        "parameters": {
            "type": "object",
            "properties": {
                "degrees": {
                    "type": "number",
                    "default": 15,
                    "description": "Degrees to rotate (default 15)",
                }
            },
        },
    },
    {
        "name": "rotate_right",
        "description": "Rotate agent right by specified degrees.",
        "parameters": {
            "type": "object",
            "properties": {
                "degrees": {
                    "type": "number",
                    "default": 15,
                    "description": "Degrees to rotate (default 15)",
                }
            },
        },
    },
    {
        "name": "move_forward",
        "description": "Move agent forward by specified distance.",
        "parameters": {
            "type": "object",
            "properties": {
                "distance": {
                    "type": "number",
                    "default": 0.1,
                    "description": "Distance in meters (default 0.1)",
                }
            },
        },
    },
    {
        "name": "move_back",
        "description": "Move agent backward by specified distance.",
        "parameters": {
            "type": "object",
            "properties": {
                "distance": {
                    "type": "number",
                    "default": 0.1,
                    "description": "Distance in meters (default 0.1)",
                }
            },
        },
    },
    {
        "name": "move_left",
        "description": "Strafe/move agent left by specified distance.",
        "parameters": {
            "type": "object",
            "properties": {
                "distance": {
                    "type": "number",
                    "default": 0.1,
                    "description": "Distance in meters (default 0.1)",
                }
            },
        },
    },
    {
        "name": "move_right",
        "description": "Strafe/move agent right by specified distance.",
        "parameters": {
            "type": "object",
            "properties": {
                "distance": {
                    "type": "number",
                    "default": 0.1,
                    "description": "Distance in meters (default 0.1)",
                }
            },
        },
    },
    {
        "name": "crouch",
        "description": "Crouch down (reduce agent height).",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "stand",
        "description": "Stand up (return to normal agent height).",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_observation",
        "description": "Get current observation describing visible objects and agent state.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "list_objects",
        "description": "List all visible objects with their properties and distances.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_position",
        "description": "Get current agent position and rotation.",
        "parameters": {"type": "object", "properties": {}},
    },
    {
        "name": "get_object_metadata",
        "description": "Get full detailed metadata for a specific object by its ID in the environment.",
        "parameters": {
            "type": "object",
            "properties": {
                "object_id": {
                    "type": "string",
                    "description": "The target object ID to inspect",
                }
            },
            "required": ["object_id"],
        },
    },
    {
        "name": "finish",
        "description": "Signal that the goal has been achieved.",
        "parameters": {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "Final message describing what was done",
                }
            },
        },
    },
]


def get_tool_schemas():
    """Return tool definitions in OpenAI function calling format"""
    return [{"type": "function", "function": tool} for tool in TOOLS]


def get_tool_names():
    """Return list of tool names"""
    return [tool["name"] for tool in TOOLS]
