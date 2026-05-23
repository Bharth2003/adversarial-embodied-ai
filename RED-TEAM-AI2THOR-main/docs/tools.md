# Tools

> **File:** `src/ai2thor_lab/tools.py`

This file defines the **complete tool/action catalogue** available to the agent. It's a list of JSON-schema tool definitions used for both LLM function calling and executor dispatch.

---

## Tool Categories

### Object Interaction
| Tool | Description | Key parameter |
|---|---|---|
| `navigate_to` | Pathfind to an object | `object_id` |
| `pickup` | Pick up an object (must be nearby) | `object_id` |
| `drop` | Drop the held object | — |
| `place_on` | Place held object on a receptacle | `receptacle_id` |
| `open` / `close` | Open or close an openable object | `object_id` |
| `toggle_on` / `toggle_off` | Turn a toggleable object on/off | `object_id` |

### State Transformations
| Tool | Description | Key parameter |
|---|---|---|
| `slice` | Slice a sliceable object | `object_id` |
| `cook` | Cook a cookable object | `object_id` |
| `fill_with_liquid` | Fill with water/coffee/wine | `object_id`, `liquid` |
| `empty_liquid` | Empty liquid from object | `object_id` |
| `clean` / `dirty` | Clean or dirty an object | `object_id` |
| `break_object` | Break a breakable object | `object_id` |
| `use_up` | Use up a consumable | `object_id` |

### Movement & Camera
| Tool | Description |
|---|---|
| `look_up` / `look_down` / `look_straight` | Camera tilt controls |
| `rotate_left` / `rotate_right` | Agent rotation (default 15°) |
| `move_forward` / `move_back` / `move_left` / `move_right` | Agent movement (default 0.1m) |
| `crouch` / `stand` | Stance changes |

### Observation
| Tool | Description |
|---|---|
| `get_observation` | Text description of current state |
| `list_objects` | List all visible objects |
| `get_position` | Current agent position/rotation |
| `get_object_metadata` | Detailed metadata for a specific object |

### Control
| Tool | Description |
|---|---|
| `finish` | Signal that the task is complete |

## Helper Functions

- `get_tool_schemas()` — wraps each tool in `{"type": "function", "function": ...}` for OpenAI-style function calling.
- `get_tool_names()` — returns just the list of tool name strings.
