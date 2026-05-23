import cv2
import json
from .agent import Agent

def print_help():
    """Print all available commands"""
    print("\n[HELP] Complete iTHOR Actions:")
    print("  MOVEMENT: 'forward [0.5]', 'back', 'left', 'right'")
    print("  ROTATION: 'turn left [45]', 'turn right [30]'")
    print("  LOOK: 'look up', 'look down', 'look straight'")
    print("  STANCE: 'crouch', 'stand'")
    print("")
    print("  BASIC: 'open fridge', 'close fridge', 'pick up knife', 'drop'")
    print("  TOGGLE: 'turn on stoveknob', 'turn off microwave', 'turn on faucet'")
    print("  PLACE: 'place on countertop', 'place pot on stoveburner'")
    print("")
    print("  LIQUID: 'fill mug with coffee', 'fill cup with water', 'empty mug'")
    print("  SLICE: 'slice potato', 'cut bread'")
    print("  COOK: 'cook potato'")
    print("  BREAK: 'break egg' (WARNING: creates shards that block movement!)")
    print("  DIRTY/CLEAN: 'dirty plate', 'clean bowl'")
    print("  USE UP: 'use up toilet paper'")
    print("")
    print("  INFO: 'look' / 'list' (list all), 'where' (position), 'metadata' (JSON dump), 'help'")

def interactive_mode(agent):
    """Interactive loop"""
    print("\niTHOR COMPLETE Control Interface")
    print("=" * 60)
    print_help()
    print("=" * 60)

    # agent.display_frame()

    while True:
        cmd = input("\n> ").strip()

        if cmd.lower() in ['quit', 'exit', 'q']:
            break

        if cmd.lower() == 'metadata':
            print("\n[METADATA] Current AI2-THOR Scene State:")
            print(json.dumps(agent.controller.last_event.metadata, indent=2))
            continue

        if cmd.lower() in ['look', 'list']:
            objects = agent.get_visible_objects()
            print("\n[OBJECTS] Visible objects:")
            for obj in sorted(objects, key=lambda x: x['distance']):
                props = []
                if obj['pickupable']: props.append("pickupable")
                if obj['receptacle']: props.append("RECEPTACLE")
                if obj['openable']: props.append("OPEN" if obj['isOpen'] else "CLOSED")
                if obj['toggleable']: props.append("ON" if obj['isToggled'] else "OFF")
                if obj['canFillWithLiquid']: props.append("fillable")
                if obj['isFilledWithLiquid']: props.append("FILLED")
                if obj['sliceable'] and not obj['isSliced']: props.append("sliceable")
                if obj['isSliced']: props.append("SLICED")
                if obj['cookable'] and not obj['isCooked']: props.append("cookable")
                if obj['isCooked']: props.append("COOKED")
                if obj['breakable'] and not obj['isBroken']: props.append("breakable")
                if obj['isBroken']: props.append("BROKEN")
                if obj['dirtyable']: props.append("DIRTY" if obj['isDirty'] else "clean")

                dist_str = f"{obj['distance']:.2f}m"
                print(f"  - {obj['name']:<20} ({dist_str:>6}) {', '.join(props)}")

            if agent.held_object:
                print(f"\n[HOLD] Holding: {(agent.held_object or '').split('|')[0]}")
            continue

        if cmd.lower() == 'where':
            meta = agent.controller.last_event.metadata['agent']
            print(f"\n[POS] x={meta['position']['x']:.2f}, y={meta['position']['y']:.2f}, z={meta['position']['z']:.2f}")
            print(f"[ROT] {meta['rotation']['y']:.1f} deg")
            print(f"[CAM] Horizon: {meta['cameraHorizon']:.1f} deg")
            continue

        if cmd.lower() == 'help':
            print_help()
            continue

        if cmd:
            agent.execute_command(cmd)

    cv2.destroyAllWindows()
    print("\n[*] Goodbye!")
