"""ERL agent for TDC tasks.

Point --agent_func_path to this file to enable ERL reflection+retry
on top of the standard ToolCallingTurn multi-turn executor.

ERL config is read from env vars (set via CLI args or training script):
  OPENRLHF_ERL_HARD_THRESHOLD, OPENRLHF_ERL_K, OPENRLHF_ERL_MEMORY,
  OPENRLHF_ERL_MAX_MEMORY, OPENRLHF_ERL_MAX_REFLECTION_TOKENS,
  OPENRLHF_CHAT_PROTOCOL
"""

from openrlhf.utils.erl_executor import ERLExecutor
from openrlhf.utils.tool_calling_turn import AgentExecutor as InnerAgentExecutor


class AgentExecutor(ERLExecutor):
    def __init__(self, **kwargs):
        super().__init__(inner_executor=InnerAgentExecutor(**kwargs))
