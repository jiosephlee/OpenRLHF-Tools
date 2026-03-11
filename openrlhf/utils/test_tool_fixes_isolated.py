import json
import sys
import os
from unittest.mock import MagicMock

# Define the repair function here to avoid imports if possible, 
# but let's try to import it first
try:
    from openrlhf.utils.chat_protocol import _repair_invalid_json_escapes
except ImportError:
    # If we can't import it, let's copy the logic for testing
    _VALID_JSON_ESC = set(['"', "\\", "/", "b", "f", "n", "r", "t", "u"])
    def _repair_invalid_json_escapes(s: str) -> str:
        out = []
        in_str = False
        i = 0
        while i < len(s):
            c = s[i]
            if not in_str:
                if c == '"':
                    in_str = True
                out.append(c)
                i += 1
                continue
            if c == '"':
                in_str = False
                out.append(c)
                i += 1
                continue
            if c == "\\":
                if i + 1 >= len(s):
                    out.append("\\\\")
                    i += 1
                    continue
                nxt = s[i + 1]
                if nxt in _VALID_JSON_ESC:
                    out.append("\\")
                    out.append(nxt)
                    i += 2
                else:
                    out.append("\\\\")
                    i += 1
                continue
            out.append(c)
            i += 1
        return "".join(out)

def test_json_repair_logic():
    # Test case: SMILES with \C (invalid escape)
    s1 = '{"smiles": "C\\C=C/C"}'
    repaired1 = _repair_invalid_json_escapes(s1)
    print(f"Original: {s1}")
    print(f"Repaired: {repaired1}")
    # Should be "C\\\\C=C/C"
    assert "C\\\\C" in repaired1
    json.loads(repaired1) # Should not raise
    
    # Test case: Valid escape \n should NOT be modified
    s2 = '{"smiles": "line1\\nline2"}'
    repaired2 = _repair_invalid_json_escapes(s2)
    print(f"Original: {s2}")
    print(f"Repaired: {repaired2}")
    assert "line1\\nline2" in repaired2
    assert "line1\\\\nline2" not in repaired2

def test_tool_calling_logic():
    # Test the logic I added to _exec_with_rdkit_log_capture
    # (Copied here for isolation)
    def mock_exec_logic(fn, arguments):
        # The logic:
        if "raw" in arguments and len(arguments) == 1:
            raw_val = arguments["raw"]
            if isinstance(raw_val, str):
                try:
                    parsed = json.loads(_repair_invalid_json_escapes(raw_val))
                    if isinstance(parsed, dict):
                        arguments = parsed
                except Exception:
                    pass
        
        smiles_arg = arguments.get("smiles", arguments.get("query_smiles", ""))
        try:
            return fn(**arguments)
        except Exception as e:
            if "got an unexpected keyword argument" in str(e) and smiles_arg:
                key = "smiles" if "smiles" in arguments else "query_smiles"
                return fn(**{key: smiles_arg})
            raise e

    def mock_tool(smiles):
        return f"MW({smiles})"

    # Test Case 1: 'raw' with bad escapes
    args1 = {"raw": '{"smiles": "C\\C"}'}
    res1 = mock_exec_logic(mock_tool, args1)
    assert res1 == "MW(C\\C)"
    print("Test Case 1 passed!")

    # Test Case 2: 'raw' as a string but valid JSON
    args2 = {"raw": '{"smiles": "CCO"}'}
    res2 = mock_exec_logic(mock_tool, args2)
    assert res2 == "MW(CCO)"
    print("Test Case 2 passed!")
    
    # Test Case 3: Unexpected keyword arg fallback
    args3 = {"smiles": "CCC", "bad_arg": "val"}
    res3 = mock_exec_logic(mock_tool, args3)
    assert res3 == "MW(CCC)"
    print("Test Case 3 passed!")

if __name__ == "__main__":
    test_json_repair_logic()
    print("JSON repair logic passed!")
    test_tool_calling_logic()
    print("Tool calling logic passed!")
