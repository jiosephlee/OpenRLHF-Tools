# Adding New Model Support to OpenRLHF

This note is for the OpenRLHF RL + tool-calling stack in this repo.

Most new models do not need special OpenRLHF code unless they introduce a new tool-calling format, need ERL reflection support, or require model-specific handling such as MoE patching, VLM loading, or quantization/dequantization workarounds.

The main integration points are:

- `openrlhf/cli/train_ppo_ray.py`
- `openrlhf/trainer/ray/vllm_engine.py`
- `openrlhf/utils/tool_calling_turn.py`
- `openrlhf/utils/chat_protocol.py`
- `openrlhf/utils/erl_executor.py`
- `openrlhf/models/actor.py`
- `openrlhf/models/model.py`
- `openrlhf/kernels/moe/patch.py`

## When code changes are needed

You usually need repo changes when one of these is true:

- the model emits a tool-calling format not covered by an existing protocol
- ERL should work and the protocol cannot inject reflections yet
- the model needs special stop strings or a new assistant-generation marker
- the model is MoE and needs grouped-GEMM patching
- the model is a VLM and needs `--is_vlm` or `--language_model_only`
- the checkpoint needs quantization cleanup or dequantization-specific handling

Current built-in protocols are:

- `glm_flash`
- `glm51`
- `intern_s1`
- `gpt_oss`
- `kimi_k2`
- `qwen3`
- `qwen3_5`

If the new model matches one of those formats exactly, most of the repo-specific work is already done.

### GLM-5 / GLM-5.1 note

GLM-5 and GLM-5.1 use the same XML-style tool-calling surface as the GLM-4.7 / GLM Flash family, but the parser needs to tolerate the `glm47` details vLLM expects:

- the function name may be immediately followed by `<arg_key>` with no newline
- zero-argument calls are valid
- multiple `<tool_call>...</tool_call>` blocks may appear in one assistant response

When serving GLM-5.1 with vLLM's OpenAI-compatible server, the recipe docs currently use:

- `--tool-call-parser glm47`
- `--reasoning-parser glm45`
- `--enable-auto-tool-choice`
- `--chat-template-content-format=string`

References:

- https://docs.vllm.ai/projects/recipes/en/latest/GLM/GLM5.html
- https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/glm47_moe_tool_parser.py

### Kimi-K2 / Kimi-K2.5 note

Kimi-K2 and Kimi-K2.5 use a sectioned tool-calling format that is different from GLM/Qwen:

- tool calls are wrapped by `<|tool_calls_section_begin|> ... <|tool_calls_section_end|>`
- each call is wrapped by `<|tool_call_begin|> ... <|tool_call_end|>`
- tool call IDs have the shape `functions.func_name:idx`
- tool results must be appended with the original `tool_call_id`

When serving Kimi-K2.5 with vLLM, the recipe docs currently use:

- `--tool-call-parser kimi_k2`
- `--reasoning-parser kimi_k2`
- `--enable-auto-tool-choice`

References:

- https://docs.vllm.ai/projects/recipes/en/latest/moonshotai/Kimi-K2.5.html
- https://github.com/vllm-project/vllm/blob/main/vllm/tool_parsers/kimi_k2_tool_parser.py

## File-by-file guide

### `openrlhf/models/actor.py`

Audit this file when the new model needs:

- custom `AutoConfig` cleanup
- special quantization handling
- special attention implementation
- MoE kernel patching
- trust-remote-code model fixes
- LoRA target-module adjustments

### `openrlhf/models/model.py`

Audit this file when the new model needs:

- custom base model resolution for reward / critic
- nonstandard value-head attachment
- special config flags before wrapping

Pay attention to:

- `AutoModel._model_mapping[type(config)]`
- `value_head_prefix`

### `openrlhf/cli/train_ppo_ray.py`

Audit this file when:

- you need a new `--chat_protocol` choice
- the model needs different default stop strings
- the model is a VLM
- the model needs new user-facing flags

Today this file already exposes:

- `--chat_protocol`
- `--vllm_stop_strings`
- `--is_vlm`
- `--language_model_only`

### `openrlhf/trainer/ray/vllm_engine.py`

Audit this file when:

- the new protocol needs a new generation marker
- the model needs special env vars inside the Ray worker
- vLLM and HF should load different paths
- hidden instruction injection must locate a different assistant-turn boundary

Important detail:

`_GENERATION_PROMPT_MARKERS` must stay in sync with the protocol's actual assistant-start marker. If you add a new protocol but do not add its marker here, hidden-instruction injection falls back to appending text at the end of the prompt.

### `openrlhf/utils/tool_calling_turn.py`

Audit this file when:

- adding a new protocol class
- selecting a different parser based on `OPENRLHF_CHAT_PROTOCOL`
- tool-call reward shaping needs model-specific behavior

Important detail:

`ToolCallingTurn.step()` uses both:

- `protocol.render_tool_feedback(...)`
- `protocol.render_tool_feedback_token_ids(...)`

If the protocol's special tokens do not survive a text -> tokenizer round trip cleanly, implement `render_tool_feedback_token_ids()` instead of relying on plain string tokenization.

### `openrlhf/utils/chat_protocol.py`

A new protocol usually needs to implement:

- `parse_assistant_text()`
- `render_tool_feedback()`
- `generation_prompt_marker`

Sometimes it also needs:

- `render_tool_feedback_token_ids()`
- `inject_reflection()`

Important details to preserve:

- Some protocols must close the assistant turn before inserting tool outputs.
- Some protocols need canonical feedback token IDs.
- Some protocols need tolerant parsing because vLLM detokenization may insert spaces or because the model emits malformed JSON.
- `generation_prompt_marker` must match the actual rendered prompt, not an approximate string.

### `openrlhf/utils/erl_executor.py`

Audit this file when ERL is enabled for a new protocol.

Current limitation:

- `_get_protocol()` only supports `intern_s1`, `qwen3`, and `qwen3_5` for reflection injection.

If the new protocol should work with ERL:

- add it to `_get_protocol()`
- implement `inject_reflection()` in the protocol class

### `openrlhf/kernels/moe/patch.py`

Audit this file if the new model is MoE and should use grouped GEMM / Unsloth-style expert patching.

If the new model has a different expert-module structure, this file may need a new detection path.

## Adding a new tool-calling protocol

### Required changes

1. Add a new `ChatProtocol` subclass in `openrlhf/utils/chat_protocol.py`.
2. Implement `parse_assistant_text()`.
3. Implement `render_tool_feedback()`.
4. Set an exact `generation_prompt_marker`.
5. If needed, implement `render_tool_feedback_token_ids()`.
6. If ERL should work, implement `inject_reflection()`.
7. Register the protocol in `openrlhf/utils/tool_calling_turn.py`.
8. Add the protocol name to `--chat_protocol` choices in `openrlhf/cli/train_ppo_ray.py`.
9. Add the generation marker to `openrlhf/trainer/ray/vllm_engine.py`.
10. If ERL should support it, register it in `openrlhf/utils/erl_executor.py`.

## Verification checklist

- tool calls parse correctly from real vLLM output
- tool feedback reopens the next assistant turn correctly
- hidden instruction insertion lands at the real generation marker
- ERL reflection injection preserves a valid prompt format
- MoE / VLM / quantized checkpoints still work after any model-specific patches

## Common failure modes

- tool calls never execute because the assistant format does not match the selected protocol
- the second turn breaks because `render_tool_feedback()` did not close and reopen turns correctly
- hidden instruction or ERL corrupts the prompt because the generation marker or injection point is wrong
