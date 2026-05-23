import json
from ai2thor_lab.safety import SafetyReviewer

def mock_llm_client(response_data):
    """Returns a mock LLM client that returns the provided data."""
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

def test_safety_prompt_generation():
    reviewer = SafetyReviewer(llm_client=lambda x, y: None)
    dummy_plan = {"task": "do something", "steps": []}
    
    prompt = reviewer._build_safety_prompt("do something", dummy_plan)
    
    assert "do something" in prompt
    assert "SAFETY RULES" in prompt
    assert "approved" in prompt

def test_safety_approved_plan():
    approved_response = {
        "approved": True,
        "reason": "Safe plan",
        "flagged_steps": []
    }
    
    reviewer = SafetyReviewer(llm_client=mock_llm_client(approved_response))
    result = reviewer.review_plan("get apple", {"steps": []})
    
    assert result["approved"] is True
    assert result["review"]["reason"] == "Safe plan"
    assert len(result["review"]["flagged_steps"]) == 0

def test_safety_rejected_plan():
    rejected_response = {
        "approved": False,
        "reason": "Dangerous actions detected",
        "flagged_steps": [2, 4]
    }
    
    reviewer = SafetyReviewer(llm_client=mock_llm_client(rejected_response))
    result = reviewer.review_plan("get apple", {"steps": []})
    
    assert result["approved"] is False
    assert result["review"]["reason"] == "Dangerous actions detected"
    assert result["review"]["flagged_steps"] == [2, 4]

def test_safety_invalid_response():
    bad_response = {"wrong_key": True}
    
    reviewer = SafetyReviewer(llm_client=mock_llm_client(bad_response))
    result = reviewer.review_plan("get apple", {"steps": []})
    
    assert result["approved"] is False
    assert "error" in result
