import importlib.util
import sys
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "data" / "tdc" / "build_v16_no_neighbor_local_reasoning_traces.py"
    spec = importlib.util.spec_from_file_location("local_reasoning_traces", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


module = _load_module()


def _item(feature: str, display_name: str, contribution: float) -> dict:
    return {
        "feature": feature,
        "feature_display_name": display_name,
        "description": display_name,
        "contribution": contribution,
        "abs_contribution": abs(contribution),
        "raw_value": 1.0,
    }


def test_select_summary_items_enforces_one_opposite_item_for_clear_calls():
    items = [
        _item("rdalert::alpha_alert", "Shared Alert", 0.90),
        _item("toxalert::alpha_alert", "Shared Alert", 0.80),
        _item("logp", "logP", 0.70),
        _item("tpsa", "TPSA", 0.60),
        _item("hbd", "HBD", -0.20),
    ]
    decision_summary = {"final_label": "B", "final_prob_pos": 0.91}

    selected = module.select_summary_items(items, decision_summary)

    assert len(selected) == 3
    assert sum(1 for item in selected if item["contribution"] < 0) == 1
    assert sum(1 for item in selected if item["feature_display_name"] == "Shared Alert") == 1


def test_select_summary_items_enforces_two_opposite_items_when_uncertain():
    items = [
        _item("logp", "logP", 0.90),
        _item("tpsa", "TPSA", 0.80),
        _item("hba", "HBA", 0.70),
        _item("hbd", "HBD", 0.60),
        _item("rotatable_bonds", "Rotatable bonds", 0.50),
        _item("aromatic_rings", "Aromatic rings", 0.40),
        _item("largest_aromatic_system", "Largest aromatic system", 0.30),
        _item("rdalert::acid_alert", "Acid Alert", -0.25),
        _item("toxalert::cation_alert", "Cation Alert", -0.20),
    ]
    decision_summary = {"final_label": "B", "final_prob_pos": 0.62}

    selected = module.select_summary_items(items, decision_summary)

    assert len(selected) == 7
    assert sum(1 for item in selected if item["contribution"] < 0) == 2
    assert [item["abs_contribution"] for item in selected] == sorted(
        [item["abs_contribution"] for item in selected],
        reverse=True,
    )


def test_render_reasoning_trace_reads_like_follow_up_turn_and_ends_with_answer():
    selected = [
        _item("toxalert::Aromatic amine (specific)", "Aromatic amine (specific)", 0.82),
        _item("rdalert::Oxygen-nitrogen_single_bond", "Oxygen-nitrogen single bond", 0.47),
        _item("aromatic_rings", "Aromatic rings", -0.03),
    ]
    decision_summary = {"base_prob_pos": 0.51, "final_label": "B", "final_prob_pos": 0.84}

    trace = module.render_reasoning_trace(selected, decision_summary, "A")

    assert trace.startswith("First, the base prior for deciding on (A)")
    assert "Analyzing the tool results, we first notice" in trace
    assert "Next," in trace
    assert "Finally," in trace
    assert "After weighing the evidence step by step" in trace
    assert trace.rstrip().endswith("Answer: (A)")
