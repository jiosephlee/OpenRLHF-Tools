import importlib.util
import json
from pathlib import Path


def _load_converter_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "scripts" / "convert_eval_sft_distill_traces_to_messages.py"
    spec = importlib.util.spec_from_file_location("convert_eval_sft_distill_traces", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


converter = _load_converter_module()


def test_build_output_record_reconstructs_glm_tool_trace_from_response():
    trace_row = {
        "task": "Bioavailability_Ma",
        "datasource": "Bioavailability_Ma",
        "smiles": "CCN(CC)c1cc(C)nc2ncnn12",
        "label": "(B)",
        "source_label": 1,
        "source_answer": "(B)",
        "trace_messages": [
            {
                "role": "assistant",
                "content": (
                    "Let me inspect the molecule.</think>"
                    "<tool_call>get_features"
                    "<arg_key>smiles</arg_key><arg_value>CCN(CC)c1cc(C)nc2ncnn12</arg_value>"
                    "<arg_key>feature_names</arg_key>"
                    "<arg_value>[\"molecular_profile\", \"ionization_and_solubility\", \"structure_and_topology\"]</arg_value>"
                    "</tool_call>"
                ),
            },
            {
                "role": "assistant",
                "content": "Answer: (B)",
            },
        ],
        "response": (
            "Let me inspect the molecule.</think>"
            "<tool_call>get_features"
            "<arg_key>smiles</arg_key><arg_value>CCN(CC)c1cc(C)nc2ncnn12</arg_value>"
            "<arg_key>feature_names</arg_key>"
            "<arg_value>[\"molecular_profile\", \"ionization_and_solubility\", \"structure_and_topology\"]</arg_value>"
            "</tool_call>\n"
            "<tool_response>"
            "{\"result\": \"ok\", \"function_name\": \"get_features\", "
            "\"arguments\": {\"smiles\": \"CCN(CC)c1cc(C)nc2ncnn12\", "
            "\"feature_names\": [\"molecular_profile\", \"ionization_and_solubility\", \"structure_and_topology\"]}}"
            "</tool_response>\n"
            "<|assistant|>\n"
            "Answer: (B)"
        ),
    }
    base_record = {
        "messages": [
            {"role": "system", "content": "You are a chemist analyzing drug molecules."},
            {"role": "user", "content": "Question text"},
        ],
        "answer": "(B)",
        "label": 1,
    }

    record = converter.build_output_record(trace_row, base_record, Path("fake.jsonl"))
    messages = record["messages"]

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool", "assistant"]
    assert messages[2]["thinking"] == "Let me inspect the molecule."
    assert messages[2]["tool_calls"][0]["function"]["name"] == "get_features"
    assert messages[2]["tool_calls"][0]["function"]["arguments"]["feature_names"] == [
        "molecular_profile",
        "ionization_and_solubility",
        "structure_and_topology",
    ]
    assert messages[3]["role"] == "tool"
    assert json.loads(messages[3]["content"])["function_name"] == "get_features"
    assert messages[4]["content"] == "Answer: (B)"
