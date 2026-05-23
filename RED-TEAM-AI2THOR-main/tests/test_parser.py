import pytest
from ai2thor_lab.parser import CommandParser

class MockAgent:
    """Mock agent to inject visible objects and state without loading AI2-THOR."""
    def __init__(self, visible_objects=None, held=None, horizon=0):
        self.objects = visible_objects or []
        self.held_object = held
        
        class MockEvent:
            metadata = {
                'agent': {
                    'cameraHorizon': horizon
                }
            }
        
        class MockController:
            last_event = MockEvent()
            
        self.controller = MockController()
        
    def get_visible_objects(self):
        return self.objects

@pytest.fixture
def empty_parser():
    """Parser with no visible objects around it."""
    return CommandParser(MockAgent())

def test_look_commands(empty_parser):
    assert empty_parser.parse_command("look up") == {'action': 'LookUp'}
    assert empty_parser.parse_command("look down") == {'action': 'LookDown'}

def test_movement_commands(empty_parser):
    assert empty_parser.parse_command("forward") == {'action': 'MoveAhead', 'moveMagnitude': 0.1}
    assert empty_parser.parse_command("move forward 1.5") == {'action': 'MoveAhead', 'moveMagnitude': 1.5}
    assert empty_parser.parse_command("back") == {'action': 'MoveBack', 'moveMagnitude': 0.1}
    assert empty_parser.parse_command("strafe left") == {'action': 'MoveLeft', 'moveMagnitude': 0.1}

def test_rotation_commands(empty_parser):
    assert empty_parser.parse_command("turn left") == {'action': 'RotateLeft', 'degrees': 15}
    assert empty_parser.parse_command("rotate right 45") == {'action': 'RotateRight', 'degrees': 45.0}

def test_stance_commands(empty_parser):
    assert empty_parser.parse_command("crouch") == {'action': 'Crouch'}
    assert empty_parser.parse_command("stand") == {'action': 'Stand'}

def test_find_object_and_interact():
    objects = [
        {'name': 'Apple', 'id': 'Apple|1', 'distance': 1.0, 'pickupable': True, 'sliceable': True, 'isSliced': False},
        {'name': 'Microwave', 'id': 'Microwave|1', 'distance': 2.0, 'toggleable': True},
        {'name': 'Mug', 'id': 'Mug|1', 'distance': 1.5, 'canFillWithLiquid': True, 'isFilledWithLiquid': False}
    ]
    parser = CommandParser(MockAgent(visible_objects=objects))
    
    # Test picking up
    res = parser.parse_command("pick up apple")
    assert res == {'action': 'PickupObject', 'objectId': 'Apple|1'}
    
    # Test slicing
    res = parser.parse_command("slice apple")
    assert res == {'action': 'SliceObject', 'objectId': 'Apple|1'}
    
    # Test toggling
    res = parser.parse_command("turn on microwave")
    assert res == {'action': 'ToggleObjectOn', 'objectId': 'Microwave|1'}
    
    # Test liquid actions
    res = parser.parse_command("fill mug with coffee")
    assert res == {'action': 'FillObjectWithLiquid', 'objectId': 'Mug|1', 'fillLiquid': 'coffee', 'filling': 'coffee'}

def test_place_receptacle():
    # Agent is holding an object (Apple|1) and wants to place it on a Table
    objects = [
        {'name': 'Table', 'id': 'Table|1', 'distance': 1.0, 'receptacle': True}
    ]
    agent = MockAgent(visible_objects=objects, held="Apple|1")
    parser = CommandParser(agent)
    
    res = parser.parse_command("place on table")
    assert res == {'action': 'PutObject', 'objectId': 'Table|1', 'placing': True, 'placing_on': 'Table'}

def test_error_handling():
    objects = [
        {'name': 'Apple', 'id': 'Apple|1', 'distance': 1.0, 'sliceable': False}
    ]
    parser = CommandParser(MockAgent(visible_objects=objects))
    
    # Missing object
    res = parser.parse_command("pick up knife")
    assert 'error' in res
    assert 'find object' in res['error'].lower() or 'not found' in res['error'].lower()
    
    # Invalid action for object attribute
    res = parser.parse_command("slice apple")
    assert 'error' in res
    assert 'cannot be sliced' in res['error'].lower()
