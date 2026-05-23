import pytest
from ai2thor_lab.plan_executor import PlanExecutor

class MockEvent:
    def __init__(self, success=True, err="", objects=None):
        self.metadata = {
            "lastActionSuccess": success, 
            "errorMessage": err,
            "objects": objects or []
        }

class MockController:
    def __init__(self, objects=None):
        # Build last_event from objects list for _build_object_index
        thor_objects = []
        for obj in (objects or []):
            thor_objects.append({
                "objectId": obj.get("id", ""),
                "objectType": obj.get("id", "").split("|")[0],
                "name": obj.get("id", "").split("|")[0],
                "visible": obj.get("visible", False),
                "distance": obj.get("distance", 10.0),
            })
        self.last_event = MockEvent(objects=thor_objects)

    def step(self, **kwargs):
        self.last_kwargs = kwargs
        if kwargs.get("action") == "FailAction":
            return MockEvent(False, "Simulated failure")
        return MockEvent()

class MockAgent:
    def __init__(self, objects=None, held=None):
        self.controller = MockController(objects=objects)
        self.objects = objects or []
        self.held_object = held
        self.executed_commands = []
        
    def get_all_objects(self):
        return self.objects
        
    def execute_command(self, cmd):
        self.executed_commands.append(cmd)
        return cmd != "fail_cmd"
        
    def get_agent_position(self):
        return {"x": 0, "z": 0, "rotation": 0}
        
    def display_frame(self):
        pass

class MockNavigator:
    def __init__(self):
        pass
    def build_reachable_map(self):
        pass
    def navigate_to_object(self, agent, target):
        return True, "Arrived"

# Patch the navigator inside PlanExecutor for testing
@pytest.fixture
def mock_executor(monkeypatch):
    from ai2thor_lab import plan_executor
    monkeypatch.setattr(plan_executor, "Navigator", lambda x: MockNavigator())
    return None # We'll instantiate locally in tests to set custom state

def test_pre_action_validation(mock_executor):
    objects = [
        {"id": "Obj1", "visible": True, "distance": 1.0, "isOpen": False},
        {"id": "ObjFar", "visible": True, "distance": 5.0},
        {"id": "ObjHidden", "visible": False, "distance": 1.0}
    ]
    agent = MockAgent(objects=objects)
    executor = PlanExecutor(agent)
    
    # Valid interactable object
    valid, msg = executor._is_interaction_target_valid("Obj1")
    assert valid is True
    
    # Missing object
    valid, msg = executor._is_interaction_target_valid("Missing")
    assert valid is False
    assert "not found" in msg.lower()
    
    # Invisible object
    valid, msg = executor._is_interaction_target_valid("ObjHidden")
    assert valid is False
    assert "not visible" in msg.lower()
    
    # Too far object
    valid, msg = executor._is_interaction_target_valid("ObjFar")
    assert valid is False
    assert "too far" in msg.lower()

def test_state_optimization_skips(mock_executor):
    objects = [
        {"id": "OpenFridge", "visible": True, "distance": 1.0, "isOpen": True},
        {"id": "MyApple", "visible": True, "distance": 1.0}
    ]
    agent = MockAgent(objects=objects, held="MyApple")
    executor = PlanExecutor(agent)
    
    # Opening an already open object should return success=True and skip
    res = executor._execute_tool("open", {"object_id": "OpenFridge"})
    assert res["success"] is True
    assert "already open" in res["message"].lower()
    
    # Picking up an already held object
    res = executor._execute_tool("pickup", {"object_id": "MyApple"})
    assert res["success"] is True
    assert "already holding" in res["message"].lower()

def test_post_action_verification(mock_executor):
    # Simulate an agent where post-action state doesn't update (failure)
    objects = [{"id": "Obj1", "visible": True, "distance": 1.0, "isOpen": False}]
    agent = MockAgent(objects=objects)
    executor = PlanExecutor(agent)
    
    # Controller step says "success", but the object metadata 'isOpen' wasn't set to True
    res = executor._execute_tool("open", {"object_id": "Obj1"})
    assert res["success"] is False
    assert "not open" in res["message"].lower()

def test_plan_execution_loop(mock_executor):
    objects = [{"id": "Apple|1", "visible": True, "distance": 1.0}]
    agent = MockAgent(objects=objects)
    executor = PlanExecutor(agent)
    
    # We patch the pickup executor logic to actually mutate the mock agent state
    # so the post-condition check passes.
    original_execute_tool = executor._execute_tool
    
    def mocked_execute_tool(tool_name, arguments):
        if tool_name == "pickup":
            agent.held_object = arguments["object_id"]
        return original_execute_tool(tool_name, arguments)
        
    executor._execute_tool = mocked_execute_tool

    plan = {
        "steps": [
            {"step": 1, "tool": "navigate_to", "arguments": {"object_id": "Apple|1"}},
            {"step": 2, "tool": "pickup", "arguments": {"object_id": "Apple|1"}},
            {"step": 3, "tool": "finish", "arguments": {}}
        ]
    }
    
    result = executor.execute_plan(plan)
    assert result["success"] is True
    assert len(result["report"]) == 3
    assert result["report"][1]["tool"] == "pickup"
    assert result["report"][1]["result"]["success"] is True
