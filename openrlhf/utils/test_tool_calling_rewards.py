import os
import sys
import asyncio
from math import isclose

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from openrlhf.tools.therapeutic_tools import v16
from openrlhf.trainer.ppo_utils.experience_maker import SamplesGenerator
from openrlhf.utils.tool_calling_turn import ToolCallingTurn
from openrlhf.utils.tool_versions import get_version, resolve_tool_metric_metadata


FEATURE_NAMES = [
    "molecular_profile",
    "ionization_and_solubility",
    "structure_and_topology",
    "alert_screening",
]


def dummy_feature_tool(smiles, feature_names):
    return ""


def dummy_neighbor_tool(smiles, task_name, feature_names=None, include_labels=True):
    return ""


class _ProtocolStub:
    def __init__(self, tool_calls):
        self._tool_calls = tool_calls

    def parse_assistant_text(self, action_text, token_ids=None):
        return {
            "tool_calls": self._tool_calls,
            "parse_method": "primary",
            "parse_failed": False,
        }

    def render_tool_feedback(self, tool_msgs):
        return "\n".join(msg["content"] for msg in tool_msgs)

    def render_tool_feedback_token_ids(self, tool_msgs):
        return []


class _TokenizerStub:
    def decode(self, token_ids, skip_special_tokens=True):
        return "".join(chr(65 + (tok % 26)) for tok in token_ids)


def _build_reward_stub(tools):
    inst = object.__new__(ToolCallingTurn)
    inst.tools = dict(tools)
    inst.tool_version = "v16"
    inst._tool_calling_reward_naive_per_call = 0.1
    inst._tool_calling_reward_feature_single = 0.1
    inst._tool_calling_reward_feature_full = 0.065
    inst._tool_calling_reward_feature_max_count = 21
    inst._tool_calling_reward_max_rewarded_calls = -1
    inst._rewarded_tool_calls_so_far = 0
    return inst


def test_v16_feature_reward_uses_actual_vocab_size():
    inst = _build_reward_stub({"get_features": dummy_feature_tool})

    reward_one = inst._compute_tool_call_reward(
        {"name": "get_features", "arguments": {"feature_names": ["molecular_profile"]}},
        "feature_aware",
    )
    reward_all = inst._compute_tool_call_reward(
        {
            "name": "get_features",
            "arguments": {
                "feature_names": [
                    "molecular_profile",
                    "ionization_and_solubility",
                    "structure_and_topology",
                    "alert_screening",
                ]
            },
        },
        "feature_aware",
    )

    assert isclose(reward_one, 0.1)
    assert isclose(reward_all, 0.065)


def test_neighbor_reward_bonus_only_requires_any_feature():
    inst = _build_reward_stub({"get_neighbors": dummy_neighbor_tool})

    reward_plain = inst._compute_tool_call_reward(
        {"name": "get_neighbors", "arguments": {"task_name": "AMES"}},
        "feature_aware",
    )
    reward_with_features = inst._compute_tool_call_reward(
        {
            "name": "get_neighbors",
            "arguments": {"task_name": "AMES", "feature_names": ["molecular_profile"]},
        },
        "feature_aware",
    )

    assert isclose(reward_plain, 0.065)
    assert isclose(reward_with_features, 0.1)


def test_feature_reward_does_not_depend_on_exact_tool_name():
    inst = _build_reward_stub({"molecule_probe": dummy_feature_tool})

    reward = inst._compute_tool_call_reward(
        {
            "name": "molecule_probe",
            "arguments": {
                "feature_names": [
                    "molecular_profile",
                    "ionization_and_solubility",
                    "structure_and_topology",
                    "alert_screening",
                ]
            },
        },
        "feature_aware",
    )

    assert isclose(reward, 0.065)


def test_neighbor_reward_does_not_depend_on_exact_tool_name():
    def renamed_neighbor_tool(smiles, feature_names=None, include_labels=True):
        return ""

    inst = _build_reward_stub({"neighbor_probe": renamed_neighbor_tool})

    reward_plain = inst._compute_tool_call_reward(
        {"name": "neighbor_probe", "arguments": {"smiles": "CCO"}},
        "feature_aware",
    )
    reward_with_features = inst._compute_tool_call_reward(
        {
            "name": "neighbor_probe",
            "arguments": {"smiles": "CCO", "feature_names": ["molecular_profile"]},
        },
        "feature_aware",
    )

    assert isclose(reward_plain, 0.065)
    assert isclose(reward_with_features, 0.1)


def test_reward_count_cap_starts_at_third_tool_call():
    inst = _build_reward_stub({"get_neighbors": dummy_neighbor_tool})
    inst._tool_calling_reward_max_rewarded_calls = 2

    reward1, rewarded1, suppressed1 = inst._compute_step_tool_calling_reward(
        [{"name": "get_neighbors", "arguments": {"task_name": "AMES"}}],
        "feature_aware",
    )
    reward2, rewarded2, suppressed2 = inst._compute_step_tool_calling_reward(
        [{"name": "get_neighbors", "arguments": {"task_name": "AMES", "feature_names": ["molecular_profile"]}}],
        "feature_aware",
    )
    reward3, rewarded3, suppressed3 = inst._compute_step_tool_calling_reward(
        [{"name": "get_neighbors", "arguments": {"task_name": "AMES"}}],
        "feature_aware",
    )

    assert isclose(reward1, 0.065)
    assert rewarded1 == 1
    assert suppressed1 == 0

    assert isclose(reward2, 0.1)
    assert rewarded2 == 1
    assert suppressed2 == 0

    assert isclose(reward3, 0.0)
    assert rewarded3 == 0
    assert suppressed3 == 1


def test_feature_request_metrics_are_recorded_by_tool_family():
    inst = _build_reward_stub(
        {
            "get_features": dummy_feature_tool,
            "get_neighbors": dummy_neighbor_tool,
        }
    )

    metrics = inst._accumulate_feature_request_metrics(
        [
            {
                "name": "get_features",
                "arguments": {
                    "feature_names": [
                        "molecular_profile",
                        "ionization_and_solubility",
                    ]
                },
            },
            {
                "name": "get_neighbors",
                "arguments": {
                    "task_name": "AMES",
                    "feature_names": ["molecular_profile"],
                },
            },
            {
                "name": "get_neighbors",
                "arguments": {"task_name": "AMES"},
            },
        ]
    )

    assert isclose(metrics["get_features_request_count"], 1.0)
    assert isclose(metrics["get_features_requested_feature_total"], 2.0)
    assert isclose(metrics["get_features_requested_feature_count_count"], 1.0)
    assert isclose(metrics["get_neighbors_request_count"], 2.0)
    assert isclose(metrics["get_neighbors_requested_feature_total"], 1.0)
    assert isclose(metrics["get_neighbors_requested_feature_count_count"], 2.0)


def test_failed_tool_turn_is_terminated_and_marked_for_discard():
    def failing_tool(smiles):
        raise ValueError(
            "SMILES 'CCCC' is not part of task 'AMES'; compare_similar_mols requires a known task molecule."
        )

    inst = object.__new__(ToolCallingTurn)
    inst.protocol = _ProtocolStub([{"name": "compare_similar_mols", "arguments": {"smiles": "CCCC"}, "id": "call_1"}])
    inst.tools = {"compare_similar_mols": failing_tool}
    inst.tool_version = "v15_neighbor_only"
    inst._task = "AMES"
    inst._current_observation_text = ""
    inst._discard_failed_tool_traces = True
    inst._enable_tool_calling_rewards = True
    inst._tool_calling_reward_until_step = -1
    inst._tool_calling_reward_mode = "feature_aware"
    inst._tool_calling_reward_naive_per_call = 0.1
    inst._tool_calling_reward_feature_single = 0.1
    inst._tool_calling_reward_feature_full = 0.065
    inst._tool_calling_reward_feature_max_count = 21
    inst._tool_calling_reward_max_rewarded_calls = -1
    inst._current_global_step = 0
    inst._total_training_steps = -1
    inst._rewarded_tool_calls_so_far = 0

    result = asyncio.run(inst.step({"action_text": "unused", "label": "(A)"}))

    assert result["done"] is True
    assert isclose(result["rewards"].item(), 0.0)
    assert isclose(result["extra_logs"]["tool_execution_failed"], 1.0)
    assert isclose(result["extra_logs"]["smiles_lookup_failed"], 1.0)
    assert isclose(result["extra_logs"]["discard_from_training"], 1.0)
    assert "is not part of task" in result["environment_feedback"]


def test_failed_tool_turn_can_continue_when_discard_is_opted_out():
    def failing_tool(smiles):
        raise ValueError("boom")

    inst = object.__new__(ToolCallingTurn)
    inst.protocol = _ProtocolStub([{"name": "compare_similar_mols", "arguments": {"smiles": "CCCC"}, "id": "call_1"}])
    inst.tools = {"compare_similar_mols": failing_tool}
    inst.tool_version = "v15_neighbor_only"
    inst._task = "AMES"
    inst._current_observation_text = ""
    inst._discard_failed_tool_traces = False
    inst._enable_tool_calling_rewards = True
    inst._tool_calling_reward_until_step = -1
    inst._tool_calling_reward_mode = "feature_aware"
    inst._tool_calling_reward_naive_per_call = 0.1
    inst._tool_calling_reward_feature_single = 0.1
    inst._tool_calling_reward_feature_full = 0.065
    inst._tool_calling_reward_feature_max_count = 21
    inst._tool_calling_reward_max_rewarded_calls = -1
    inst._current_global_step = 0
    inst._total_training_steps = -1
    inst._rewarded_tool_calls_so_far = 0

    result = asyncio.run(inst.step({"action_text": "unused", "label": "(A)"}))

    assert result["done"] is False
    assert isclose(result["rewards"].item(), 0.0)
    assert isclose(result["extra_logs"]["tool_execution_failed"], 1.0)
    assert isclose(result["extra_logs"]["discard_from_training"], 0.0)


def test_process_response_into_experience_drops_flagged_trace():
    gen = object.__new__(SamplesGenerator)
    gen.tokenizer = _TokenizerStub()
    gen._write_hidden_prompt_stripping_audit = lambda **kwargs: None

    response = {
        "observation_tokens": [1, 2, 3],
        "action_ranges": [(1, 3)],
        "rollout_log_probs": None,
        "reward": 0.0,
        "scores": 0.0,
        "prompt": "prompt",
        "label": "(A)",
        "extra_logs": {
            "discard_from_training": 1.0,
            "tool_execution_failed": 1.0,
        },
    }

    exp = gen._process_response_into_experience(response, prompt_max_len=16, max_new_tokens=16)

    assert exp is None


def test_v15_metric_mapping_counts_fixed_surface_feature_requests_without_avg_denominator():
    def v15_feature_tool(smiles):
        return ""

    def v15_neighbor_tool(smiles):
        return ""

    inst = _build_reward_stub(
        {
            "get_mol_properties_and_fg": v15_feature_tool,
            "compare_similar_mols": v15_neighbor_tool,
        }
    )
    inst.tool_version = "v15"

    metrics = inst._accumulate_feature_request_metrics(
        [
            {"name": "get_mol_properties_and_fg", "arguments": {"smiles": "CCO"}},
            {"name": "compare_similar_mols", "arguments": {"smiles": "CCO"}},
        ]
    )

    assert isclose(metrics["get_features_request_count"], 1.0)
    assert isclose(metrics["get_features_requested_feature_total"], 0.0)
    assert isclose(metrics["get_features_requested_feature_count_count"], 0.0)
    assert isclose(metrics["get_neighbors_request_count"], 1.0)
    assert isclose(metrics["get_neighbors_requested_feature_total"], 0.0)
    assert isclose(metrics["get_neighbors_requested_feature_count_count"], 0.0)


def test_v15_registry_declares_metric_endpoints():
    feature_metadata = resolve_tool_metric_metadata("v15", "get_mol_properties_and_fg")
    neighbor_metadata = resolve_tool_metric_metadata("v15", "compare_similar_mols")

    assert feature_metadata == {"endpoint": "features", "count_request_metric": True}
    assert neighbor_metadata == {"endpoint": "neighbors", "count_request_metric": True}


def test_v15_variant_registries_declare_single_endpoint_surfaces():
    no_neighbor_cfg = get_version("v15_no_neighbor")
    neighbor_only_cfg = get_version("v15_neighbor_only")

    assert [schema["function"]["name"] for schema in no_neighbor_cfg["basic_schemas"]] == ["get_mol_properties_and_fg"]
    assert no_neighbor_cfg["task_specific_map"] == {}
    assert set(no_neighbor_cfg["callables"]) == {"get_mol_properties_and_fg"}
    assert resolve_tool_metric_metadata("v15_no_neighbor", "get_mol_properties_and_fg") == {
        "endpoint": "features",
        "count_request_metric": True,
    }

    assert [schema["function"]["name"] for schema in neighbor_only_cfg["basic_schemas"]] == ["compare_similar_mols"]
    assert neighbor_only_cfg["task_specific_map"] == {}
    assert set(neighbor_only_cfg["callables"]) == {"compare_similar_mols"}
    assert resolve_tool_metric_metadata("v15_neighbor_only", "compare_similar_mols") == {
        "endpoint": "neighbors",
        "count_request_metric": True,
    }


def test_v16_registry_uses_single_dynamic_neighbor_tool():
    cfg = get_version("v16")
    basic_names = [schema["function"]["name"] for schema in cfg["basic_schemas"]]

    assert basic_names == ["get_features", "get_neighbors"]
    assert cfg["task_specific_map"] == {}
    assert set(cfg["callables"]) == {"get_features", "get_neighbors"}


def test_v16_feature_schema_description_is_clean_and_non_repetitive():
    description = v16.GET_FEATURES_TOOL["function"]["description"]
    feature_desc = v16.GET_FEATURES_TOOL["function"]["parameters"]["properties"]["feature_names"]["description"]

    assert "groups of properties" not in description
    assert "such as a (1)" not in description
    assert "molecular profile" in description
    assert "molecular_profile covers" in feature_desc


def test_v16_neighbor_feature_output_includes_query_neighbor_delta_summary():
    output = v16.get_neighbors(
        "Nc1cccc([N+](=O)[O-])c1CO",
        "AMES",
        feature_names=["molecular_profile", "structure_and_topology", "alert_screening"],
    )

    assert "molecular_profile:" in output
    assert "neighbor=" in output
    assert "delta=" in output
    assert "query=" not in output
    assert "Added vs query" not in output
    assert "Alert categories (query)" not in output


if __name__ == "__main__":
    test_v16_feature_reward_uses_actual_vocab_size()
    test_neighbor_reward_bonus_only_requires_any_feature()
    test_feature_reward_does_not_depend_on_exact_tool_name()
    test_neighbor_reward_does_not_depend_on_exact_tool_name()
    test_reward_count_cap_starts_at_third_tool_call()
    test_feature_request_metrics_are_recorded_by_tool_family()
    test_failed_tool_turn_is_terminated_and_marked_for_discard()
    test_failed_tool_turn_can_continue_when_discard_is_opted_out()
    test_process_response_into_experience_drops_flagged_trace()
    test_v15_metric_mapping_counts_fixed_surface_feature_requests_without_avg_denominator()
    test_v15_registry_declares_metric_endpoints()
    test_v15_variant_registries_declare_single_endpoint_surfaces()
    test_v16_registry_uses_single_dynamic_neighbor_tool()
    test_v16_feature_schema_description_is_clean_and_non_repetitive()
    test_v16_neighbor_feature_output_includes_query_neighbor_delta_summary()
    print("tool calling reward tests passed")
