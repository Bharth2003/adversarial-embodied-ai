import re
from ai2thor_lab.action_knowledge import (
    ACTION_KNOWLEDGE,
    get_action_for_command,
    validate_action,
)


class CommandParser:
    def __init__(self, agent):
        self.agent = agent

    def find_object(self, command, exclude_held=False):
        """Find object in command - FIXED for full IDs and toggle commands"""
        objects = self.agent.get_visible_objects()
        cmd_lower = command.lower()

        # Don't split "turn on X" or "turn off X" commands
        if any(cmd_lower.startswith(prefix) for prefix in
               ['turn on ', 'turn off ', 'toggle on ', 'toggle off ',
                'switch on ', 'switch off ']):
            # Extract object name after the action phrase
            for prefix in ['turn on ', 'turn off ', 'toggle on ', 'toggle off ',
                          'switch on ', 'switch off ']:
                if cmd_lower.startswith(prefix):
                    search_part = cmd_lower[len(prefix):]
                    break
        else:
            # Remove "on X" part for place commands
            search_part = cmd_lower.split(' on ')[0] if ' on ' in cmd_lower else cmd_lower
            # Remove "with X" part if exists
            search_part = search_part.split(' with ')[0] if ' with ' in search_part else search_part

        # 1. Try full ID match first (high precision)
        if '|' in search_part:
            for obj in objects:
                if obj['id'].lower() in search_part.lower() or search_part.lower() in obj['id'].lower():
                    if exclude_held and self.agent.held_object and obj['id'] == self.agent.held_object:
                        continue
                    return obj

        # 2. Fallback to name-based match
        for obj in objects:
            if obj['name'].lower() in search_part:
                if exclude_held and self.agent.held_object and obj['id'] == self.agent.held_object:
                    continue
                return obj
        return None

    def find_receptacle(self, command):
        """Find receptacle after 'on' keyword or nearest"""
        objects = self.agent.get_visible_objects()

        if ' on ' in command.lower():
            target_part = command.lower().split(' on ', 1)[1].strip()

            # 1. Try full ID match first
            if '|' in target_part:
                for obj in objects:
                    if obj['receptacle'] and (obj['id'].lower() in target_part.lower() or target_part.lower() in obj['id'].lower()):
                        return obj

            # 2. Fallback to name-based match
            for obj in objects:
                if obj['receptacle'] and obj['name'].lower() in target_part:
                    return obj

        # Find nearest receptacle
        receptacles = [obj for obj in objects if obj['receptacle']]
        if receptacles:
            return min(receptacles, key=lambda x: x['distance'])
        return None

    def parse_command(self, command):
        """Parse a natural-language command into an AI2-THOR action.

        Movement, look, stance, and rotation commands are handled directly.
        Object interactions are resolved via ACTION_KNOWLEDGE (data-driven).
        """
        cmd_lower = command.lower()
        
        # Extract numbers and liquids
        numbers = re.findall(r'\d+\.?\d*', command)
        value = float(numbers[0]) if numbers else None

        # Extract liquid type (water, coffee, wine)
        liquid = None
        for liq in ['water', 'coffee', 'wine']:
            if liq in cmd_lower:
                liquid = liq
                break

        # === LOOK CONTROLS ===
        if 'look up' in cmd_lower:
            return {'action': 'LookUp'}
        if 'look down' in cmd_lower:
            return {'action': 'LookDown'}
        if 'look straight' in cmd_lower or 'look forward' in cmd_lower:
            current_horizon = self.agent.controller.last_event.metadata['agent']['cameraHorizon']
            steps = abs(int(current_horizon / 30))
            if current_horizon > 0:
                return {'action': 'LookUp', 'repeat': steps}
            elif current_horizon < 0:
                return {'action': 'LookDown', 'repeat': steps}
            return {'action': 'Pass'}

        # === STANCE ===
        if 'crouch' in cmd_lower or 'duck' in cmd_lower:
            return {'action': 'Crouch'}
        if 'stand' in cmd_lower and 'look' not in cmd_lower:
            return {'action': 'Stand'}

        # === ROTATION ===
        if 'rotate left' in cmd_lower or 'turn left' in cmd_lower:
            degrees = value if value else 15
            return {'action': 'RotateLeft', 'degrees': degrees}
        if 'rotate right' in cmd_lower or 'turn right' in cmd_lower:
            degrees = value if value else 15
            return {'action': 'RotateRight', 'degrees': degrees}

        # === MOVEMENT ===
        if re.search(r'\bforward\b', cmd_lower) and not re.search(r'\blook\b', cmd_lower):
            distance = value if value else 0.1
            return {'action': 'MoveAhead', 'moveMagnitude': distance}
        if re.search(r'\bback\b', cmd_lower):
            distance = value if value else 0.1
            return {'action': 'MoveBack', 'moveMagnitude': distance}
        if re.search(r'\b(left|strafe left)\b', cmd_lower) and not re.search(r'\b(turn|rotate)\b', cmd_lower):
            distance = value if value else 0.1
            return {'action': 'MoveLeft', 'moveMagnitude': distance}
        if re.search(r'\b(right|strafe right)\b', cmd_lower) and not re.search(r'\b(turn|rotate)\b', cmd_lower):
            distance = value if value else 0.1
            return {'action': 'MoveRight', 'moveMagnitude': distance}

        # ================================================================
        # OBJECT INTERACTIONS — resolved via ACTION_KNOWLEDGE
        # ================================================================
        action_name = get_action_for_command(cmd_lower)
        if action_name is None:
            return {'error': 'Could not determine action'}

        knowledge = ACTION_KNOWLEDGE[action_name]

        # --- Special case: drop (no object lookup needed) ---
        if action_name == "drop":
            return {'action': knowledge['ai2thor_action']}

        # --- Special case: put/place (needs receptacle) ---
        if action_name == "put":
            if not self.agent.held_object:
                return {'error': 'Not holding any object to place'}
            receptacle = self.find_receptacle(command)
            if receptacle:
                return {'action': knowledge['ai2thor_action'],
                        'objectId': receptacle['id'],
                        'placing': True,
                        'placing_on': receptacle['name']}
            return {'error': 'No receptacle found to place object'}

        # --- General case: find object, validate, build result ---
        obj = self.find_object(command)
        if not obj:
            return {'error': 'Object not found'}

        valid, reason = validate_action(action_name, obj)
        if not valid:
            return {'error': reason}

        result = {'action': knowledge['ai2thor_action'], 'objectId': obj['id']}

        # Attach liquid parameter for fill actions
        if action_name == "fill":
            liquid_type = liquid or 'water'
            result['fillLiquid'] = liquid_type
            result['filling'] = liquid_type

        return result
