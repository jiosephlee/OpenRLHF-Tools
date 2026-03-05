"""ERL (Experiential Reinforcement Learning) executor.

Wraps any AgentExecutorBase to add prompt-level gated reflection and retry.
For hard prompts (avg reward < threshold), generates structured reflections
from failed attempts and retries with reflection-augmented prompts.

Loaded via --agent_func_path pointing to an agent .py that exports this
class (or a subclass) as ``AgentExecutor``.  See ``erl_tdc_agent.py`` for
an example.

Based on: Experiential Reinforcement Learning (Shi et al., Feb 2026)
https://arxiv.org/abs/2602.13949

STATUS: EXPERIMENTAL — not yet tested end-to-end.
"""

import asyncio
import os
import re
from collections import defaultdict, deque
from typing import List

from openrlhf.utils.agent import AgentExecutorBase
from openrlhf.utils.logging_utils import init_logger

logger = init_logger(__name__)

# TDC task names for memory grouping
_TDC_TASK_NAMES = [
    "AMES",
    "BBB_Martins",
    "Bioavailability_Ma",
    "HIA_Hou",
    "PAMPA_NCATS",
    "Pgp_Broccatelli",
    "CYP2C9",
    "CYP2D6",
    "CYP3A4",
    "SARSCoV2_3CLPro",
    "SARSCoV2_Vitro",
    "Carcinogens",
    "hERG",
    "ClinTox",
    "DILI",
    "Skin_Reaction",
    "Tox21",
]


class ERLExecutor(AgentExecutorBase):
    """Wraps an existing executor with ERL reflection+retry logic.

    For each prompt batch (via ``execute_batch``):
    1. Run n samples via the inner executor (standard path)
    2. If avg reward >= hard_threshold → return n results (easy prompt)
    3. If avg reward < hard_threshold (hard prompt):
       a. Generate k diverse reflections from different failed attempts
       b. Inject each reflection into the prompt via ChatProtocol.inject_reflection()
       c. Run k retry episodes via the inner executor
       d. Optionally store successful reflections in per-task memory
       e. Return n + k results (variable group size)

    Config is read from env vars (set by CLI / training script):
      - OPENRLHF_ERL_HARD_THRESHOLD (float)
      - OPENRLHF_ERL_K (int, default 4)
      - OPENRLHF_ERL_MEMORY (0/1, default 0)
      - OPENRLHF_ERL_MAX_MEMORY (int, default 5)
      - OPENRLHF_ERL_MAX_REFLECTION_TOKENS (int, default 512)
      - OPENRLHF_CHAT_PROTOCOL (str, default "intern_s1")
    """

    def __init__(self, inner_executor: AgentExecutorBase):
        self.inner = inner_executor
        self.hard_threshold = float(os.environ.get("OPENRLHF_ERL_HARD_THRESHOLD", "0.2"))
        self.erl_k = int(os.environ.get("OPENRLHF_ERL_K", "4"))
        self._chat_protocol_name = os.environ.get("OPENRLHF_CHAT_PROTOCOL", "intern_s1")

        # Lazy-init protocol and memory
        self._protocol = None
        self._memory = defaultdict(deque)
        self._max_memory = int(os.environ.get("OPENRLHF_ERL_MAX_MEMORY", "5"))

    # ------------------------------------------------------------------
    # AgentExecutorBase interface — single-sample passthrough
    # ------------------------------------------------------------------

    async def execute(self, prompt, label, sampling_params, max_length, hf_tokenizer, llm_engine, **kwargs):
        """Passthrough to inner executor (non-ERL path)."""
        return await self.inner.execute(
            prompt=prompt,
            label=label,
            sampling_params=sampling_params,
            max_length=max_length,
            hf_tokenizer=hf_tokenizer,
            llm_engine=llm_engine,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Batch entry point — called by LLMRayActor.generate_responses
    # when executor has execute_batch().  Returns variable-size list.
    # ------------------------------------------------------------------

    async def execute_batch(
        self,
        prompt: str,
        label: str,
        sampling_params,
        max_length: int,
        num_samples: int,
        hf_tokenizer,
        llm_engine,
        log_trajectory: bool = False,
    ) -> list:
        """Generate samples with prompt-level ERL gating and k diverse retries."""

        # Step 1: Generate all n samples (standard path)
        tasks = [
            self.inner.execute(
                prompt=prompt,
                label=label,
                sampling_params=sampling_params,
                max_length=max_length,
                hf_tokenizer=hf_tokenizer,
                llm_engine=llm_engine,
                log_trajectory=log_trajectory,
            )
            for _ in range(num_samples)
        ]
        results = await asyncio.gather(*tasks)

        # Step 2: Check prompt difficulty
        rewards = [r.get("reward", 0) for r in results]
        avg_reward = sum(rewards) / len(rewards) if rewards else 0

        if avg_reward >= self.hard_threshold:
            for r in results:
                r.setdefault("extra_logs", {})["erl_gated"] = 0
            return results

        # Step 3: Hard prompt — generate k diverse reflections
        failed_results = [r for r in results if r.get("reward", 0) < 1.0]
        if not failed_results:
            for r in results:
                r.setdefault("extra_logs", {})["erl_gated"] = 0
            return results

        max_refl_tokens = int(os.environ.get("OPENRLHF_ERL_MAX_REFLECTION_TOKENS", "512"))
        from vllm import SamplingParams as _SamplingParams

        refl_params = _SamplingParams(
            temperature=sampling_params.temperature,
            top_p=sampling_params.top_p,
            max_tokens=max_refl_tokens,
        )

        # Generate k reflections in parallel, each from a different failed attempt
        reflection_tasks = []
        for i in range(self.erl_k):
            exemplar = failed_results[i % len(failed_results)]
            attempt_text = hf_tokenizer.decode(exemplar["observation_tokens"], skip_special_tokens=False)
            refl_prompt_text = self._build_reflection_prompt(prompt, attempt_text, hf_tokenizer)
            refl_token_ids = hf_tokenizer(refl_prompt_text, add_special_tokens=False, return_tensors="pt")[
                "input_ids"
            ][0].tolist()
            reflection_tasks.append(llm_engine.generate(refl_token_ids, refl_params))

        reflection_outputs = await asyncio.gather(*reflection_tasks)
        reflections = [o.outputs[0].text for o in reflection_outputs]

        # Step 4: Run k retries in parallel, each with a different reflection
        protocol = self._get_protocol(hf_tokenizer)
        retry_tasks = []
        for reflection_text in reflections:
            retry_prompt = protocol.inject_reflection(prompt, reflection_text)
            retry_tasks.append(
                self.inner.execute(
                    prompt=retry_prompt,
                    label=label,
                    sampling_params=sampling_params,
                    max_length=max_length,
                    hf_tokenizer=hf_tokenizer,
                    llm_engine=llm_engine,
                    log_trajectory=False,
                )
            )
        retry_results = await asyncio.gather(*retry_tasks)

        # Step 5: Memory update (if enabled) — store first successful reflection
        use_memory = os.environ.get("OPENRLHF_ERL_MEMORY", "0") == "1"
        if use_memory:
            for rr, refl in zip(retry_results, reflections):
                if rr.get("reward", 0) >= 1.0:
                    self._update_memory(prompt, refl)
                    break

        # Step 6: Tag and return all results (original n + k retries)
        for r in results:
            r.setdefault("extra_logs", {})["erl_gated"] = 0
            r["extra_logs"]["erl_hard_prompt"] = 1
        for rr in retry_results:
            rr.setdefault("extra_logs", {})["erl_gated"] = 1
            rr["extra_logs"]["erl_hard_prompt"] = 1
            rr["extra_logs"]["erl_r2"] = rr.get("reward", 0)
            # Generic distillation signal: any executor can set this to
            # request SFT loss on this experience's action tokens.
            if rr.get("reward", 0) >= 1.0:
                rr["extra_logs"]["distill"] = 1

        logger.info(
            "[ERL] Hard prompt: avg_r1=%.2f, r2_rewards=%s, group_size=%d",
            avg_reward,
            [rr.get("reward", 0) for rr in retry_results],
            len(results) + len(retry_results),
        )

        return results + retry_results

    # ------------------------------------------------------------------
    # Reflection prompt construction
    # ------------------------------------------------------------------

    def _build_reflection_prompt(self, original_prompt: str, attempt_text: str, hf_tokenizer) -> str:
        """Build reflection prompt with task context + tool names."""
        tool_names = self._extract_tool_names(original_prompt)
        tool_names_str = ", ".join(tool_names) if tool_names else "various chemistry tools"

        # Truncate attempt to last ~2000 chars to fit in context
        attempt_truncated = attempt_text

        # Get memory text if enabled
        memory_text = ""
        if os.environ.get("OPENRLHF_ERL_MEMORY", "0") == "1":
            memory_text = self._get_memory(original_prompt)

        memory_section = ""
        if memory_text:
            memory_section = f"## Lessons from Similar Tasks\n{memory_text}\n\n"

        user_content = (
            f"## Original Task\n{original_prompt}\n\n"
            f"## Available Tools\n{tool_names_str}\n\n"
            f"## Previous Attempt (INCORRECT)\n{attempt_truncated}\n\n"
            f"{memory_section}"
            "Reflect on this failed attempt and provide guidance for a retry. "
            "Consider the following:\n"
            "- Which tools from the available set were NOT used but could provide critical insight?\n"
            "- What connections between molecular properties (e.g., lipophilicity, charge state, "
            "functional groups, steric effects) were missed?\n"
            "- Perform a deeper structure-activity relationship (SAR) analysis: what structural "
            "features of this molecule are most relevant to the classification task?\n"
            "- What alternative interpretation of the tool results could lead to a different conclusion?\n\n"
            "Be specific and actionable (5-10 sentences), synthesizing insights that would have helped you succeed when you first attempted the task."
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a medicinal chemistry expert analyzing a failed molecular classification attempt. "
                    "Your goal is to generate diverse, explorative strategies for the retry."
                ),
            },
            {"role": "user", "content": user_content},
        ]
        return hf_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_protocol(self, hf_tokenizer):
        """Lazily initialize the chat protocol for reflection injection."""
        if self._protocol is None:
            from openrlhf.utils.chat_protocol import InternS1Protocol, Qwen3Protocol

            if self._chat_protocol_name == "intern_s1":
                self._protocol = InternS1Protocol(hf_tokenizer)
            elif self._chat_protocol_name == "qwen3":
                self._protocol = Qwen3Protocol(hf_tokenizer)
            else:
                raise NotImplementedError(
                    f"ERL inject_reflection not implemented for chat_protocol={self._chat_protocol_name!r}. "
                    "Currently only 'intern_s1' and 'qwen3' are supported."
                )
        return self._protocol

    @staticmethod
    def _extract_tool_names(prompt: str) -> List[str]:
        """Extract tool function names from the formatted prompt."""
        names = re.findall(r'"name"\s*:\s*"([^"]+)"', prompt)
        seen = set()
        return [n for n in names if not (n in seen or seen.add(n))]

    def _task_key(self, prompt: str) -> str:
        """Extract task identifier from prompt for memory grouping."""
        prompt_lower = prompt.lower()
        for task in _TDC_TASK_NAMES:
            if task.lower() in prompt_lower:
                return task
        return "__default__"

    def _update_memory(self, prompt: str, reflection: str):
        task_key = self._task_key(prompt)
        mem = self._memory[task_key]
        mem.append(reflection)
        if len(mem) > self._max_memory:
            mem.popleft()

    def _get_memory(self, prompt: str) -> str:
        task_key = self._task_key(prompt)
        entries = self._memory.get(task_key, [])
        if not entries:
            return ""
        return "\n".join(f"- {e}" for e in entries)
