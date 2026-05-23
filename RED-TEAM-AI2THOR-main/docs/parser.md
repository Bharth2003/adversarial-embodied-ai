# Command Parser

> **File:** `src/ai2thor_lab/parser.py`

The `CommandParser` translates **natural-language text commands** (typed by a human in interactive mode) into AI2-THOR controller action dicts. It's a rule-based parser — no LLM involved.

---

## How It Works

`parse_command(command)` walks through a priority-ordered chain of keyword matches:

1. **Look controls** — `"look up"`, `"look down"`, `"look straight"`
2. **Stance** — `"crouch"`, `"stand"`
3. **Rotation** — `"turn left 45"`, `"rotate right"`
4. **Movement** — `"forward 0.5"`, `"back"`, `"left"`, `"right"`
5. **Drop** — `"drop"`
6. **Place** — `"place on countertop"`, `"put pot on stoveburner"`
7. **Liquid** — `"fill mug with coffee"`, `"empty cup"`
8. **Slice / Cook / Break** — `"slice potato"`, `"cook egg"`, `"break vase"`
9. **Dirty / Clean** — `"dirty plate"`, `"clean bowl"`
10. **Use up** — `"use up toilet paper"`
11. **Toggle** — `"turn on faucet"`, `"switch off microwave"`
12. **Basic interactions** — `"pick up knife"`, `"open fridge"`, `"close cabinet"`

## Object Resolution

`find_object(command)` matches the command text against currently **visible** objects by checking if the object's name appears in the command string (case-insensitive). Special handling for toggle commands to avoid splitting on the word "on" (e.g. *"turn on stoveknob"* shouldn't split at "on").

`find_receptacle(command)` finds the target surface for placement by looking for a receptacle name after the word "on" in the command.

## Validation

The parser validates preconditions before returning an action:
- Can't fill something that isn't fillable or is already filled
- Can't slice something that isn't sliceable or is already sliced
- Can't dirty something that isn't dirtyable or is already dirty
- etc.

Returns `{"error": "..."}` on failure, which the `Agent` prints to the user.

## Usage

Only used in **interactive mode** (`cli.py`). The LLM-based pipelines bypass this parser entirely and call the controller directly.
