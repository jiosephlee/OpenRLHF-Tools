"""Tool-calling turn for GRPO training — single-class agent.

Implements ``AgentInstanceBase`` for multi-turn tool-calling rollouts.

The initial prompt arrives **already formatted** by ``--apply_chat_template``
in the preprocessing step.  ``reset()`` is a passthrough; only ``step()``
does real work (parse tool calls, execute, produce bridge text via
``protocol.render_tool_feedback``).
"""

import json
import os
import re
import sys
import types
import torch
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Extern tools — import modules directly, bypassing Intern-S1-recipe's
# tools/__init__.py which hard-depends on molgpka (not installed).
# We pre-register an empty ``tools`` package in sys.modules so that
# ``from tools.RDKit_tools import ...`` resolves each submodule without
# ever executing __init__.py.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_INTERN_S1_ROOT = _PROJECT_ROOT / "Intern-S1-recipe"
assert (_INTERN_S1_ROOT / "tools").is_dir(), (
    f"Intern-S1-recipe/tools not found at {_INTERN_S1_ROOT}/tools. "
    f"Run: git submodule update --init Intern-S1-recipe"
)
if str(_INTERN_S1_ROOT) not in sys.path:
    sys.path.insert(0, str(_INTERN_S1_ROOT))
if "tools" not in sys.modules:
    _pkg = types.ModuleType("tools")
    _pkg.__path__ = [str(_INTERN_S1_ROOT / "tools")]
    _pkg.__package__ = "tools"
    sys.modules["tools"] = _pkg

from tools.RDKit_tools import (
    RDKIT_BASIC_OPENAI_TOOLS,
    TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP,
    # basic
    get_molecular_weight,
    get_exact_molecular_weight,
    get_heavy_atom_count,
    get_mol_logp,
    get_tpsa,
    get_hbd,
    get_hba,
    get_num_rotatable_bonds,
    get_fraction_csp3,
    get_mol_mr,
    get_ring_count,
    get_num_aromatic_rings,
    get_formal_charge,
    get_qed,
    get_num_heteroatoms,
    # task-specific
    get_labute_asa,
    get_max_abs_partial_charge,
    get_min_abs_partial_charge,
    get_max_estate_index,
    get_min_estate_index,
    get_num_aromatic_atoms,
    get_fraction_aromatic_atoms,
    get_num_positive_charge_atoms,
    get_num_negative_charge_atoms,
    get_num_aliphatic_rings,
    get_num_saturated_rings,
    get_num_heterocycles,
    get_num_aromatic_heterocycles,
    get_num_aliphatic_heterocycles,
    get_num_saturated_heterocycles,
    get_num_amide_bonds,
    get_bertz_ct,
    get_balaban_j,
    get_ipc,
    get_hall_kier_alpha,
    get_kappa1,
    get_kappa2,
    get_kappa3,
    get_num_atom_stereo_centers,
    get_num_unspecified_atom_stereo_centers,
)
from tools.AccFG import AccFG_OPENAI_TOOLS, cached_describe_high_level_fg_fragments
from tools.standardize_tools import STANDARDIZE_OPENAI_TOOLS, remove_salts

try:
    from tools.ePSA_3D import get_3d_exposed_polar_surface, SASA_OPENAI_TOOLS
except ImportError:
    get_3d_exposed_polar_surface = None
    SASA_OPENAI_TOOLS = []

BASIC_TOOLS = (
    RDKIT_BASIC_OPENAI_TOOLS
    + AccFG_OPENAI_TOOLS
    + STANDARDIZE_OPENAI_TOOLS
    + SASA_OPENAI_TOOLS
)

# All callable tools (everything from get_function_by_name minus pKa).
_TOOL_CALLABLES: Dict[str, Callable] = {
    "describe_high_level_fg_fragments": cached_describe_high_level_fg_fragments,
    "get_molecular_weight": get_molecular_weight,
    "get_exact_molecular_weight": get_exact_molecular_weight,
    "get_heavy_atom_count": get_heavy_atom_count,
    "get_mol_logp": get_mol_logp,
    "get_tpsa": get_tpsa,
    "get_hbd": get_hbd,
    "get_hba": get_hba,
    "get_num_rotatable_bonds": get_num_rotatable_bonds,
    "get_fraction_csp3": get_fraction_csp3,
    "get_labute_asa": get_labute_asa,
    "get_mol_mr": get_mol_mr,
    "get_ring_count": get_ring_count,
    "get_num_aromatic_rings": get_num_aromatic_rings,
    "get_formal_charge": get_formal_charge,
    "get_qed": get_qed,
    "get_num_heteroatoms": get_num_heteroatoms,
    "get_max_abs_partial_charge": get_max_abs_partial_charge,
    "get_min_abs_partial_charge": get_min_abs_partial_charge,
    "get_max_estate_index": get_max_estate_index,
    "get_min_estate_index": get_min_estate_index,
    "get_num_aromatic_atoms": get_num_aromatic_atoms,
    "get_fraction_aromatic_atoms": get_fraction_aromatic_atoms,
    "get_num_positive_charge_atoms": get_num_positive_charge_atoms,
    "get_num_negative_charge_atoms": get_num_negative_charge_atoms,
    "get_num_aliphatic_rings": get_num_aliphatic_rings,
    "get_num_saturated_rings": get_num_saturated_rings,
    "get_num_heterocycles": get_num_heterocycles,
    "get_num_aromatic_heterocycles": get_num_aromatic_heterocycles,
    "get_num_aliphatic_heterocycles": get_num_aliphatic_heterocycles,
    "get_num_saturated_heterocycles": get_num_saturated_heterocycles,
    "get_num_amide_bonds": get_num_amide_bonds,
    "get_bertz_ct": get_bertz_ct,
    "get_balaban_j": get_balaban_j,
    "get_ipc": get_ipc,
    "get_hall_kier_alpha": get_hall_kier_alpha,
    "get_kappa1": get_kappa1,
    "get_kappa2": get_kappa2,
    "get_kappa3": get_kappa3,
    "get_num_atom_stereo_centers": get_num_atom_stereo_centers,
    "get_num_unspecified_atom_stereo_centers": get_num_unspecified_atom_stereo_centers,
    "remove_salts": remove_salts,
}
if get_3d_exposed_polar_surface is not None:
    _TOOL_CALLABLES["get_3d_exposed_polar_surface"] = get_3d_exposed_polar_surface

from openrlhf.utils.agent import AgentInstanceBase, MultiTurnAgentExecutor
from openrlhf.utils.chat_protocol import GLMFlashProtocol, InternS1Protocol


class ToolCallingTurn(AgentInstanceBase):
    """One trajectory of a tool-calling agent.

    Each instance owns its own conversation history and is created fresh by
    ``MultiTurnAgentExecutor.execute`` for every rollout.

    The prompt is expected to arrive already chat-templated (via
    ``--apply_chat_template`` in the data preprocessing).  The protocol is
    only used for two things:
      - ``parse_assistant_text``: extract tool calls from vLLM output
      - ``render_tool_feedback``: produce the bridge text between turns
    """

    def __init__(self, hf_tokenizer=None):
        # ---- tokenizer (needed by protocol parsers) ----
        if hf_tokenizer is not None:
            self.tokenizer = hf_tokenizer
        else:
            model_path = os.environ.get("OPENRLHF_MODEL_PATH")
            if not model_path:
                raise ValueError("OPENRLHF_MODEL_PATH environment variable must be set")
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # ---- protocol (parse + feedback only, not initial rendering) ----
        protocol_name = os.environ.get("OPENRLHF_CHAT_PROTOCOL", "glm_flash")
        if protocol_name == "intern_s1":
            self.protocol = InternS1Protocol(self.tokenizer)
        else:
            self.protocol = GLMFlashProtocol(self.tokenizer)

        # ---- tool callables ----
        # Register ALL tools (basic + task-specific) so the agent can execute
        # any tool the model calls. The prompt/chat template controls which
        # tools the model *sees*; this controls which it can *execute*.
        self.tools: Dict[str, Callable] = {}
        all_tool_specs = list(BASIC_TOOLS)
        for task_tools in TDC_RDKIT_SPECIFIC_OPENAI_TOOLS_MAP.values():
            all_tool_specs.extend(task_tools)
        seen = set()
        for tool_spec in all_tool_specs:
            if isinstance(tool_spec, dict) and "function" in tool_spec:
                func_name = tool_spec["function"]["name"]
                if func_name not in seen:
                    seen.add(func_name)
                    func = _TOOL_CALLABLES.get(func_name)
                    if func:
                        self.tools[func_name] = func

    # ------------------------------------------------------------------
    # AgentInstanceBase interface
    # ------------------------------------------------------------------

    async def reset(self, states: Dict[str, Any], **kwargs) -> Dict[str, str]:
        """Passthrough — prompt is already chat-templated by preprocessing."""
        return {"observation": states.get("observation", "")}

    async def step(self, state_dict: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Parse tool calls, execute, return upstream-contract dict."""
        action_text = state_dict["action_text"]
        label = state_dict.get("label", "")
        action = self.protocol.parse_assistant_text(action_text)
        tool_calls = action.get("tool_calls", [])

        if tool_calls:
            # Execute tools and build result dicts
            tool_msgs = []
            for tc in tool_calls:
                result = await self._execute_tool(tc)
                tool_msgs.append({"name": tc.get("name", ""), "content": result})

            # Bridge text: close assistant turn + tool responses + open next turn
            feedback = self.protocol.render_tool_feedback(tool_msgs)
            return {
                "environment_feedback": feedback,
                "rewards": torch.tensor(0.0),
                "done": False,
                "scores": 0.0,
                "extra_logs": {"tool_call_count": len(tool_calls)},
            }

        # No tool calls → final answer
        reward = self._compute_reward(action.get("content", ""), label)
        return {
            "environment_feedback": "",
            "rewards": torch.tensor(reward),
            "done": True,
            "scores": reward,
            "extra_logs": {"tool_call_count": 0},
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _execute_tool(self, tool_call: Dict[str, Any]) -> str:
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        if tool_name not in self.tools:
            return json.dumps({
                "error": f"Unknown tool: {tool_name}",
                "available_tools": list(self.tools.keys()),
            })
        try:
            result = self.tools[tool_name](**arguments)
            return json.dumps({
                "result": result,
                "function_name": tool_name,
                "arguments": arguments,
            })
        except Exception as e:
            return json.dumps({
                "error": str(e),
                "function_name": tool_name,
                "arguments": arguments,
            })

    _ANSWER_RE = re.compile(r"Answer\s*:\s*\(?\s*([A-Za-z])\s*\)?")

    def _compute_reward(self, generated_text: str, label: Optional[str]) -> float:
        """Reward = 1.0 iff the model's Answer: (X) after </think> matches the label."""
        if not label:
            return 0.0

        # Only consider text after the closing </think> tag
        think_end = generated_text.find("</think>")
        if think_end == -1:
            return 0.0
        answer_region = generated_text[think_end:]

        match = self._ANSWER_RE.search(answer_region)
        if not match:
            return 0.0

        pred = match.group(1).upper()

        # Extract letter from label too (handles "A", "(A)", "Answer: (A)", etc.)
        label_match = self._ANSWER_RE.search(label)
        if label_match:
            gold = label_match.group(1).upper()
        else:
            # Bare letter like "A" or "(A)"
            gold = label.strip().strip("()").upper()

        return 1.0 if pred == gold else 0.0


# ---------------------------------------------------------------------------
# Executor (required name for vllm_engine._load_agent_executor)
# ---------------------------------------------------------------------------
class AgentExecutor(MultiTurnAgentExecutor):
    def __init__(self):
        super().__init__(ToolCallingTurn)


__all__ = ["ToolCallingTurn", "AgentExecutor"]
