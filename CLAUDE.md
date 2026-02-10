# OpenRLHF Tool-Calling Support

## Overview

This implementation extends OpenRLHF with multi-turn tool-calling support for Group Relative Policy Optimization (GRPO) training. Agents can interact with external tools over multiple turns while maintaining proper token-level masking for policy gradients.

**Key Features:**
- ✅ Multi-turn agent-based rollouts with tool execution
- ✅ Token-level masking (only LLM actions contribute to loss, observations excluded)
- ✅ Clean abstraction layer (AgentSession + ChatProtocol)
- ✅ GLM Flash XML tool format support
- ✅ Extensible protocol system for new formats

## Architecture

### Core Components

#### 1. AgentSession (`openrlhf/utils/agent_session.py`)
Coordinates multi-turn conversations with inline tool execution and reward computation.

**Key Methods:**
```python
class AgentSession:
    async def initialize(self, prompt: str) -> str
        # Reset history, format with tools, return prompt

    async def step(self, action_text: str, label: str) -> dict
        # Parse action, execute tools, compute reward
        # Returns: {"feedback": str, "reward": float, "done": bool}
```

**Simplifications:**
- No heavyweight Environment/RewardPipeline abstractions
- Inline tool execution via simple dict: `{"tool_name": callable}`
- Inline reward computation (placeholder for actual reward model)

#### 2. ChatProtocol (`openrlhf/utils/chat_protocol.py`)
Abstract interface for model-specific formats.

**Key Methods:**
```python
class ChatProtocol(ABC):
    def render_messages(self, messages, tools, add_generation_prompt) -> str
        # Format messages with tool schemas

    def parse_assistant_text(self, text: str) -> Dict[str, Any]
        # Parse tool calls from LLM output
```

**GLMFlashProtocol:**
- Tool format: `<tool_call>func_name<arg_key>key</arg_key><arg_value>value</arg_value></tool_call>`
- Uses vLLM's official parser (with regex fallback)
- Supports manual (fast) and auto (robust) prompt construction modes

#### 3. ToolCallAgent (`openrlhf/utils/tool_calling_agent.py`)
Agent implementation using AgentSession and ChatProtocol.

**Clean Design:**
- 160 lines (vs 350+ in monolithic version)
- Uses MultiTurnAgentExecutor with factory pattern (zero-arg init)
- Delegates all logic to session and protocol

```python
class ToolCallAgent(AgentInstanceBase):
    def __init__(self):
        protocol = GLMFlashProtocol(tokenizer)
        self.session = AgentSession(protocol, tools, system_prompt)

    async def reset(self, states):
        return {"observation": await self.session.initialize(states["observation"])}

    async def step(self, state_dict):
        result = await self.session.step(state_dict["action_text"], state_dict["label"])
        return {
            "environment_feedback": result["feedback"],
            "rewards": torch.tensor(result["reward"]),
            "done": result["done"]
        }
```

### Multi-Turn Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    GRPO Training Loop                           │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│               Experience Maker (Rollout Generation)             │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│                  vLLM Engine (LLMRayActor)                      │
│  • Sets environment variables (MODEL_PATH, PROMPT_MODE, etc.)   │
│  • Loads AgentExecutor from agent_func_path                     │
└─────────────────────────────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              MultiTurnAgentExecutor (agent.py)                  │
│  • Orchestrates multi-turn loop                                 │
│  • Tracks action_ranges (only LLM tokens)                       │
│  • Computes rollout_log_probs                                   │
└─────────────────────────────────────────────────────────────────┘
                             ↓
         ┌────────────────────────────────────┐
         │                                    │
         ↓                                    ↓
┌─────────────────┐              ┌──────────────────────┐
│  ToolCallAgent  │──────────────│   AgentSession       │
│  • reset()      │              │   • initialize()     │
│  • step()       │              │   • step()           │
└─────────────────┘              │   • _execute_tool()  │
                                 │   • _compute_reward()│
                                 └──────────────────────┘
                                            ↓
                                 ┌──────────────────────┐
                                 │   ChatProtocol       │
                                 │   • render_messages()│
                                 │   • parse_text()     │
                                 └──────────────────────┘
```

### Token-Level Masking

**Critical Feature:** Only LLM-generated action tokens contribute to policy loss.

```
Trajectory: [prompt_tokens | action_1 | observation_1 | action_2 | observation_2 | final_answer]
              ├─ Prompt: system + user question (0-N)
              ├─ Action 1: tool call (N to M) ← In action_ranges
              ├─ Observation 1: tool response (M to K) ← NOT in action_ranges
              ├─ Action 2: tool call (K to L) ← In action_ranges
              ├─ Observation 2: tool response (L to P) ← NOT in action_ranges
              └─ Final: answer (P to Q) ← In action_ranges

action_ranges = [(N, M), (K, L), (P, Q)]  # Only actions!
action_mask = [0...0, 1...1, 0...0, 1...1, 0...0, 1...1]  # Binary mask
```

**GRPO Loss Computation:**
```python
loss_mask = action_mask * attention_mask  # Only LLM tokens have mask=1
actor_loss = -(log_probs * advantages * loss_mask).sum() / loss_mask.sum()
```

This ensures observations don't affect policy gradients while allowing the LLM to condition on them.

## Usage

### Quick Start

1. **Prepare your tool functions** (in external module):
```python
# therapeutic-tuning/tools/__init__.py
BASIC_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_qed",
            "description": "Calculate drug-likeness score",
            "parameters": {
                "type": "object",
                "properties": {
                    "smiles": {"type": "string", "description": "SMILES string"}
                },
                "required": ["smiles"]
            }
        }
    }
]

def calculate_qed(smiles: str) -> float:
    # Your implementation
    return 0.85
```

2. **Run training:**
```bash
bash examples/scripts/train_grpo_tool_calling.sh /path/to/model /path/to/data
```

### CLI Arguments

**Tool-Calling Specific:**
- `--agent_func_path`: Path to agent implementation (e.g., `openrlhf/utils/tool_calling_agent.py`)
- `--agent_max_steps`: Max turns per episode (default: 5, recommend: 20-40 for complex tasks)
- `--vllm_stop_strings`: Stop generation tokens (e.g., `"</tool_call>"`)
- `--prompt_construction_mode`: Prompt mode - `"manual"` (fast) or `"auto"` (robust)

**Example:**
```bash
python -m openrlhf.cli.train_ppo_ray \
    --pretrain /path/to/glm-flash-model \
    --agent_func_path openrlhf/utils/tool_calling_agent.py \
    --agent_max_steps 40 \
    --vllm_stop_strings "</tool_call>" \
    --prompt_construction_mode manual \
    --advantage_estimator dr_grpo \
    --n_samples_per_prompt 8 \
    --dynamic_filtering \
    --dynamic_filtering_reward_range 0.2 0.8 \
    # ... other GRPO args
```

### Environment Variables

Set automatically by vllm_engine.py:
- `OPENRLHF_MODEL_PATH`: Model path for tokenizer (set from `--pretrain`)
- `OPENRLHF_PROMPT_CONSTRUCTION_MODE`: Prompt mode (set from CLI arg)
- `OPENRLHF_MAX_STEPS`: Max steps (set from `--agent_max_steps`)

Additional recommended:
- `VLLM_NO_USAGE_STATS=1`: Disable vLLM telemetry
- `VLLM_DISABLE_TELEMETRY=1`: Disable vLLM telemetry

## Prompt Construction Modes

### Manual Mode (Default) - Fast
String concatenation with hardcoded format:
```python
environment_feedback = (
    f"<|observation|>\n"
    f"<tool_response>{result}</tool_response>\n"
    f"<|assistant|>\n"
)
```

**Pros:** ⚡ Minimal overhead, direct control
**Cons:** ❌ Brittle, model-specific format

### Auto Mode - Robust
Uses tokenizer's chat template:
```python
messages_with_tool = messages + [
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "content": result}
]
environment_feedback = tokenizer.apply_chat_template(
    messages_with_tool, tools=BASIC_TOOLS, add_generation_prompt=True
)
```

**Pros:** ✅ Robust, model-agnostic, handles edge cases
**Cons:** 🐢 Slower (re-processes history each turn)

**Recommendation:** Use manual for production, auto for development/testing.

## Extending with New Protocols

### Add Qwen3 Protocol (JSON Format)

1. **Create protocol class:**
```python
# openrlhf/utils/chat_protocol.py

class Qwen3Protocol(ChatProtocol):
    """Qwen3 JSON format: <tool_call>\n{"name": "...", "arguments": {...}}\n</tool_call>"""

    TOOL_CALL_REGEX = re.compile(r"<tool_call>\s*(?P<body>.*?)\s*</tool_call>", re.DOTALL)

    def parse_assistant_text(self, text: str) -> Dict[str, Any]:
        match = self.TOOL_CALL_REGEX.search(text)
        if match:
            payload = json.loads(match.group("body"))
            return {
                "content": "",
                "tool_calls": [{
                    "name": payload["name"],
                    "arguments": payload["arguments"]
                }]
            }
        return {"content": text, "tool_calls": []}
```

2. **Use in agent:**
```python
protocol_name = os.environ.get("OPENRLHF_CHAT_PROTOCOL", "glm_flash")
if protocol_name == "qwen3":
    protocol = Qwen3Protocol(tokenizer)
elif protocol_name == "glm_flash":
    protocol = GLMFlashProtocol(tokenizer)
```

3. **Set environment variable:**
```bash
export OPENRLHF_CHAT_PROTOCOL=qwen3
```

## Debugging

### Test Parser Directly
```python
from openrlhf.utils.chat_protocol import GLMFlashProtocol
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("path/to/model", trust_remote_code=True)
protocol = GLMFlashProtocol(tokenizer)

text = "<tool_call>calculate_qed<arg_key>smiles</arg_key><arg_value>CCO</arg_value></tool_call>"
action = protocol.parse_assistant_text(text)
print(action)
# {'content': '', 'tool_calls': [{'name': 'calculate_qed', 'arguments': {'smiles': 'CCO'}}]}
```
## Testing

### Unit Tests
```bash
# Test AgentSession
pytest tests/test_agent_session.py -v

# Test ChatProtocol
pytest tests/test_chat_protocol.py -v
```

### Integration Test (Small-Scale)
```bash
python -m openrlhf.cli.train_ppo_ray \
    --agent_func_path openrlhf/utils/tool_calling_agent.py \
    --max_samples 10 \
    # ... other args
```

Validate:
- ✅ Ray logs show agent initialization
- ✅ action_ranges tracked correctly
- ✅ Rewards computed (check wandb)
- ✅ No crashes in 10 samples

## Contributing

To add new features:
1. Create feature branch: `git checkout -b feature/your-feature`
2. Implement changes (maintain abstraction layer)
3. Add tests (unit + integration)
4. Update this documentation
5. Submit pull request

---

**Last Updated:** 2026-02-09
**Implemented By:** Claude Sonnet 4.5
**Base Version:** OpenRLHF (latest main branch)
