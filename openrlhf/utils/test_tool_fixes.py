import json
import sys
import os

# Add the project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from openrlhf.utils.chat_protocol import GPTOSSProtocol
from openrlhf.utils.tool_calling_turn import _exec_with_rdkit_log_capture

class MockTokenizer:
    def decode(self, token_ids, skip_special_tokens=False):
        # Very crude mock
        return "to=functions.get_molecular_weight <|message|>{\"smiles\": \"C\\C=C/C\"} <|call|>"
    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]

def test_repair_json():
    protocol = GPTOSSProtocol(MockTokenizer())
    
    # Simulate a case where valid JSON fails due to backslashes in SMILES
    # The regex fallback should use _repair_invalid_json_escapes
    token_ids = [1, 2, 3]
    raw_text = "to=functions.get_molecular_weight <|message|>{\"smiles\": \"C\\C=C/C\"}"
    
    result = protocol._regex_fallback_parse(token_ids, raw_text)
    print(f"Regex fallback result: {result}")
    assert len(result["tool_calls"]) == 1
    assert result["tool_calls"][0]["arguments"]["smiles"] == "C\\C=C/C"

def test_tool_fallback():
    # Mock function that doesn't take 'raw'
    def mock_tool(smiles):
        return f"MW of {smiles} is 100"
    
    # Case 1: arguments has 'raw' which is a JSON string
    args1 = {"raw": "{\"smiles\": \"CCO\"}"}
    err1, res1 = _exec_with_rdkit_log_capture(mock_tool, args1, "mock_tool")
    print(f"Tool fallback 1: err={err1}, res={res1}")
    assert err1 == ""
    assert "MW of CCO is 100" in res1
    
    # Case 2: arguments has 'raw' which is NOT JSON, but we have a fallback
    # Actually if it's not JSON, 'arguments' remains {"raw": "..."}
    # Then mock_tool(**arguments) fails with "unexpected keyword argument 'raw'"
    # Our fallback should catch this and try calling with smiles if it was found.
    # Wait, in _exec_with_rdkit_log_capture, if it's not JSON, smiles_arg will be ""
    # Let's test the "unexpected keyword argument" fallback.
    
    def mock_tool_2(smiles):
        return "Success"
        
    args2 = {"smiles": "CCO", "extra_bad_arg": "val"}
    err2, res2 = _exec_with_rdkit_log_capture(mock_tool_2, args2, "mock_tool_2")
    print(f"Tool fallback 2: err={err2}, res={res2}")
    assert err2 == ""
    assert "Success" in res2

if __name__ == "__main__":
    try:
        test_repair_json()
        print("test_repair_json passed!")
    except Exception as e:
        print(f"test_repair_json failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        test_tool_fallback()
        print("test_tool_fallback passed!")
    except Exception as e:
        print(f"test_tool_fallback failed: {e}")
        import traceback
        traceback.print_exc()
