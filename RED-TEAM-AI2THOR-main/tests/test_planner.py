import json
from ai2thor_lab.planner import Planner

def mock_llm_client(response_data):
    """Returns a mock LLM client that simply returns the provided data."""
    def client(messages, tools):
        return {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(response_data)
                    }
                }
            ]
        }
    return client

def test_planner_prompt_generation():
    planner = Planner(llm_client=None)
    scene_meta = {
        "scene": "FloorPlan1",
        "objects": [
            {"id": "Apple|1", "name": "Apple", "pickupable": True, "sliceable": True},
            {"id": "Fridge|1", "name": "Fridge", "receptacle": True, "openable": True, "isOpen": False}
        ]
    }
    
    prompt = planner._build_planning_prompt("get the apple from the fridge", scene_meta)
    
    # Verify key elements are in the prompt
    assert "get the apple from the fridge" in prompt
    assert "FloorPlan1" in prompt
    assert "Apple|1" in prompt
    assert "Fridge|1" in prompt
    assert "pickupable" in prompt
    assert "receptacle" in prompt
    assert "isClosed" in prompt

def test_planner_valid_plan():
    # Mock a successful plan generation
    expected_plan = {
        "task": "pick up apple",
        "steps": [
            {"step": 1, "tool": "navigate_to", "arguments": {"object_id": "Apple|1"}, "reason": "move"},
            {"step": 2, "tool": "pickup", "arguments": {"object_id": "Apple|1"}, "reason": "pick up"},
            {"step": 3, "tool": "finish", "arguments": {}, "reason": "done"}
        ]
    }
    
    planner = Planner(llm_client=mock_llm_client(expected_plan))
    result = planner.generate_plan("pick up apple", {"objects": []})
    
    assert result["success"] is True
    assert result["plan"] == expected_plan
    assert len(result["plan"]["steps"]) == 3

def test_planner_invalid_json():
    # Mock an LLM returning bad JSON or missing steps
    bad_plan = {"task": "pick up apple", "wrong_key": []}
    planner = Planner(llm_client=mock_llm_client(bad_plan))
    
    result = planner.generate_plan("pick up apple", {"objects": []})
    assert result["success"] is False
    assert "error" in result
    assert "missing 'steps'" in result["error"].lower()
