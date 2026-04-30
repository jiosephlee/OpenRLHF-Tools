import json
import random
import re
from collections import defaultdict

from torch.utils.data import Dataset
from tqdm import tqdm

from openrlhf.utils.tool_versions import filter_schemas_by_selectors, get_version

_KNN_PSEUDO_LABEL_RE = re.compile(r"pseudo label.*?KNN prediction is \(([A-Z])\)", re.IGNORECASE)


def _infer_tool_schema_style(chat_template):
    """Infer the tool schema shape expected by a tokenizer chat template.

    Some templates, notably GPT-OSS Harmony, expect each entry in ``tools`` to
    be wrapped as ``{"type": "function", "function": {...}}`` and will fail
    on the flatter OpenAI-style schema shape ``{"type": "function", "name":
    ..., "parameters": ...}``.
    """
    if not isinstance(chat_template, str) or not chat_template:
        return None
    if "tool.function" in chat_template or "tool['function']" in chat_template or 'tool["function"]' in chat_template:
        return "wrapped"
    if "tool.name" in chat_template or "tool['name']" in chat_template or 'tool["name"]' in chat_template:
        return "flat"
    return None


def _normalize_tool_schema(tool, schema_style):
    """Convert a tool schema between flat and wrapped forms when needed."""
    if not isinstance(tool, dict) or schema_style is None:
        return tool

    if schema_style == "wrapped":
        if isinstance(tool.get("function"), dict):
            return tool
        if tool.get("type") == "function" and "name" in tool:
            return {
                "type": "function",
                "function": {k: v for k, v in tool.items() if k != "type"},
            }
        return tool

    if schema_style == "flat":
        function_spec = tool.get("function")
        if tool.get("type") == "function" and isinstance(function_spec, dict) and "name" in function_spec:
            flattened = {"type": "function", **function_spec}
            for key, value in tool.items():
                if key not in ("type", "function"):
                    flattened[key] = value
            return flattened
        return tool

    return tool


def _normalize_tool_schemas_for_template(tools, schema_style):
    if not tools or schema_style is None:
        return tools
    return [_normalize_tool_schema(tool, schema_style) for tool in tools]


def interleave_indices_by_datasource(indices, datasources_list, seed):
    """Reorder indices so each datasource is evenly distributed across the sequence.

    Args:
        indices: list of integer indices into datasources_list
        datasources_list: list where datasources_list[i] gives the datasource name for index i
        seed: random seed for reproducible within-group shuffling

    Returns:
        Reordered list of indices with even spacing per datasource.
    """
    # Group indices by datasource
    groups = defaultdict(list)
    for idx in indices:
        groups[datasources_list[idx]].append(idx)

    if len(groups) <= 1:
        return list(indices)

    # Shuffle within each group (reproducible)
    rng = random.Random(seed)
    for group_indices in groups.values():
        rng.shuffle(group_indices)

    # Assign evenly-spaced positions
    n = len(indices)
    positioned = []
    for ds_name in sorted(groups.keys()):  # sorted for determinism
        group_indices = groups[ds_name]
        stride = n / len(group_indices)
        for j, idx in enumerate(group_indices):
            positioned.append((j * stride, ds_name, idx))

    # Sort by position, break ties by datasource name
    positioned.sort(key=lambda x: (x[0], x[1]))
    return [idx for _, _, idx in positioned]


def preprocess_data(
    data,
    input_template=None,
    input_key="input",
    label_key=None,
    apply_chat_template=None,
    tools_map=None,
    tool_schema_style=None,
) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if tools_map is not None:
            # Per-task tool lookup: data["task"] → tool list, fallback to __default__
            task = data.get("task", "__default__")
            task_tools = tools_map.get(task)
            default_tools = tools_map.get("__default__", [])
            if task_tools is None:
                kwargs["tools"] = _normalize_tool_schemas_for_template(default_tools, tool_schema_style)
            elif isinstance(task_tools, dict) and "_extend_default" in task_tools:
                # Resolve tool refs: strings are looked up in _tool_defs, dicts are inline schemas
                tool_defs = tools_map.get("_tool_defs", {})
                extras = [tool_defs[t] if isinstance(t, str) else t for t in task_tools["_extend_default"]]
                kwargs["tools"] = _normalize_tool_schemas_for_template(default_tools + extras, tool_schema_style)
            else:
                kwargs["tools"] = _normalize_tool_schemas_for_template(task_tools, tool_schema_style)
        prompt = apply_chat_template(chat, **kwargs)
    else:
        prompt = data[input_key]
        if isinstance(prompt, list):
            # Messages format without --apply_chat_template (e.g. agent-based training
            # where the agent handles chat template application with tool schemas).
            # Serialize as JSON so the agent can parse and merge with its own system prompt.
            prompt = json.dumps(prompt, ensure_ascii=False)
        if input_template:
            prompt = input_template.format(prompt)

    # for Reinforced Fine-tuning
    label = "" if label_key is None else data[label_key]
    return prompt, label


class PromptDataset(Dataset):
    """
    Dataset for PPO model

    Args:
        dataset: dataset for PPO model
        tokenizer: tokenizer for PPO model
        max_length: max length of input
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        strategy,
        input_template=None,
        enable_phase_curriculum: bool = False,
        dataset_key: str = "prompt_data",
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.enable_phase_curriculum = enable_phase_curriculum
        self.dataset_key = dataset_key

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template
        self.apply_chat_template = apply_chat_template
        self.tool_schema_style = _infer_tool_schema_style(getattr(self.tokenizer, "chat_template", None))
        self.input_key = input_key
        self.label_key = label_key

        # Load per-task tool schemas: {task_name: [tool_schemas], "__default__": [...]}
        tools_map = None
        tdc_tools_path = getattr(self.strategy.args, "tdc_tools", None)
        if tdc_tools_path:
            with open(tdc_tools_path) as f:
                tools_map = json.load(f)
        self.tools_map = tools_map
        self.tool_version = getattr(self.strategy.args, "tool_version", None)
        self.phase_policy = None
        if self.tool_version:
            self.phase_policy = get_version(self.tool_version).get("phase_policy")
        self.static_phase_curriculum_enabled = bool(self.enable_phase_curriculum and self.phase_policy and apply_chat_template)
        self.late_phase_hidden_instruction = None
        self.phase_split_index = None

        dataset_items = list(dataset)
        if self.static_phase_curriculum_enabled and len(dataset_items) > 1:
            rng = random.Random(getattr(self.strategy.args, "seed", 42))
            rng.shuffle(dataset_items)
            self.phase_split_index = len(dataset_items) // 2
            late_phase = (self.phase_policy.get("phases") or [None, None])[1]
            if late_phase is not None:
                self.late_phase_hidden_instruction = late_phase.get("hidden_instruction")
        else:
            self.phase_split_index = len(dataset_items)

        # Load external KNN pseudo-label mapping for prompts that lack inline pseudo labels.
        # Format: {task: {smiles: {"pseudo_label": "(A)", ...}}}
        knn_pl_map = None
        knn_pl_path = getattr(self.strategy.args, "knn_pseudo_labels_path", None)
        if knn_pl_path:
            with open(knn_pl_path) as f:
                knn_pl_map = json.load(f)

        self.prompts = []
        self.labels = []
        self.datasources = []
        self.knn_pseudo_labels = []  # KNN pseudo-label extracted from prompt text, or None
        self.raw_records = []
        self.late_phase_prompt_flags = []
        self.prompt_refs = []
        for idx, data in enumerate(tqdm(dataset_items, desc="Preprocessing data", disable=not self.strategy.is_rank_0())):
            late_phase_prompt = bool(self.static_phase_curriculum_enabled and idx >= self.phase_split_index)
            prompt, label = self._preprocess_record(
                data,
                input_template=input_template,
                input_key=input_key,
                label_key=label_key,
                apply_chat_template=apply_chat_template,
                late_phase_prompt=late_phase_prompt,
            )
            self.prompts.append(prompt)
            self.labels.append(label)
            self.datasources.append(data.get("datasource", "default"))
            self.raw_records.append(data)
            self.late_phase_prompt_flags.append(late_phase_prompt)
            self.prompt_refs.append(f"{self.dataset_key}:{idx}")
            # Use external mapping if provided (avoids regex over long prompts),
            # otherwise fall back to inline regex extraction.
            if knn_pl_map is not None:
                task = data.get("task") or data.get("datasource", "")
                smiles = data.get("smiles", "")
                entry = knn_pl_map.get(task, {}).get(smiles, {})
                pl = entry.get("pseudo_label")
                # Normalize "(A)" → "A" to match inline regex format
                if pl and pl.startswith("(") and pl.endswith(")"):
                    pl = pl[1:-1]
                self.knn_pseudo_labels.append(pl)
            else:
                m = _KNN_PSEUDO_LABEL_RE.search(prompt)
                self.knn_pseudo_labels.append(m.group(1) if m else None)

        if getattr(self.strategy.args, "curriculum_balanced", False):
            self._apply_curriculum_balancing(getattr(self.strategy.args, "seed", 42))
        self._rebuild_prompt_ref_index()

    def _apply_curriculum_balancing(self, seed):
        """Reorder samples so each datasource is evenly distributed across training."""
        if len(set(self.datasources)) <= 1:
            return

        order = interleave_indices_by_datasource(
            list(range(len(self.prompts))), self.datasources, seed
        )
        self.prompts = [self.prompts[i] for i in order]
        self.labels = [self.labels[i] for i in order]
        self.datasources = [self.datasources[i] for i in order]
        self.knn_pseudo_labels = [self.knn_pseudo_labels[i] for i in order]
        self.raw_records = [self.raw_records[i] for i in order]
        self.late_phase_prompt_flags = [self.late_phase_prompt_flags[i] for i in order]
        self.prompt_refs = [self.prompt_refs[i] for i in order]
        self._rebuild_prompt_ref_index()

    def _rebuild_prompt_ref_index(self):
        self.prompt_ref_to_idx = {prompt_ref: idx for idx, prompt_ref in enumerate(self.prompt_refs)}

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return (
            idx,
            self.datasources[idx],
            self.prompts[idx],
            self.labels[idx],
            self.knn_pseudo_labels[idx],
            self.late_phase_prompt_flags[idx],
            self.prompt_refs[idx],
        )

    def collate_fn(self, item_list):
        indices = []
        datasources = []
        prompts = []
        labels = []
        knn_pseudo_labels = []
        late_phase_prompt_flags = []
        prompt_refs = []
        for idx, datasource, prompt, label, knn_pl, late_phase_prompt, prompt_ref in item_list:
            indices.append(idx)
            datasources.append(datasource)
            prompts.append(prompt)
            labels.append(label)
            knn_pseudo_labels.append(knn_pl)
            late_phase_prompt_flags.append(late_phase_prompt)
            prompt_refs.append(prompt_ref)

        return indices, datasources, prompts, labels, knn_pseudo_labels, late_phase_prompt_flags, prompt_refs

    def get_sample_by_prompt_ref(self, prompt_ref):
        idx = self.prompt_ref_to_idx[prompt_ref]
        return {
            "idx": idx,
            "prompt_ref": prompt_ref,
            "datasource": self.datasources[idx],
            "prompt": self.prompts[idx],
            "label": self.labels[idx],
            "knn_pseudo_label": self.knn_pseudo_labels[idx],
            "late_phase_prompt": self.late_phase_prompt_flags[idx],
            "raw_record": self.raw_records[idx],
        }

    def iter_prompt_samples(self):
        for prompt_ref in self.prompt_refs:
            yield self.get_sample_by_prompt_ref(prompt_ref)

    def _resolve_tools_for_record(self, record, late_phase_prompt: bool = False):
        if self.tools_map is None:
            return None

        task = record.get("task", "__default__")
        task_tools = self.tools_map.get(task)
        default_tools = self.tools_map.get("__default__", [])
        if task_tools is None:
            selected_tools = list(default_tools)
        elif isinstance(task_tools, dict) and "_extend_default" in task_tools:
            tool_defs = self.tools_map.get("_tool_defs", {})
            extras = [tool_defs[t] if isinstance(t, str) else t for t in task_tools["_extend_default"]]
            selected_tools = list(default_tools) + extras
        else:
            selected_tools = list(task_tools)

        if not self.static_phase_curriculum_enabled:
            return selected_tools
        phases = self.phase_policy.get("phases") or []
        phase_index = 1 if late_phase_prompt and len(phases) > 1 else 0
        phase_spec = phases[phase_index] if phases else None
        if phase_spec is None:
            return selected_tools
        selectors = phase_spec.get("tool_selectors")
        if not selectors:
            return selected_tools
        return filter_schemas_by_selectors(selected_tools, selectors)

    def _preprocess_record(
        self,
        data,
        *,
        input_template,
        input_key,
        label_key,
        apply_chat_template,
        late_phase_prompt: bool,
    ):
        if apply_chat_template:
            chat = data[input_key]
            if isinstance(chat, str):
                chat = [{"role": "user", "content": chat}]
            kwargs = dict(tokenize=False, add_generation_prompt=True)
            tools = self._resolve_tools_for_record(data, late_phase_prompt=late_phase_prompt)
            if tools:
                kwargs["tools"] = _normalize_tool_schemas_for_template(tools, self.tool_schema_style)
            prompt = apply_chat_template(chat, **kwargs)
            label = "" if label_key is None else data[label_key]
            return prompt, label
        return preprocess_data(
            data,
            input_template,
            input_key,
            label_key,
            apply_chat_template,
            tools_map=self.tools_map,
            tool_schema_style=self.tool_schema_style,
        )


class PromptPoolDataset(Dataset):
    """Materialized prompt pool used for structured episode phases."""

    def __init__(self, samples, *, dataset_key: str, late_phase_hidden_instruction=None) -> None:
        super().__init__()
        self.dataset_key = dataset_key
        self.late_phase_hidden_instruction = late_phase_hidden_instruction
        self.prompts = []
        self.labels = []
        self.datasources = []
        self.knn_pseudo_labels = []
        self.raw_records = []
        self.late_phase_prompt_flags = []
        self.prompt_refs = []

        for sample in samples:
            self.prompts.append(sample["prompt"])
            self.labels.append(sample["label"])
            self.datasources.append(sample["datasource"])
            self.knn_pseudo_labels.append(sample.get("knn_pseudo_label"))
            self.raw_records.append(sample.get("raw_record"))
            self.late_phase_prompt_flags.append(bool(sample.get("late_phase_prompt", False)))
            self.prompt_refs.append(sample["prompt_ref"])

        self._rebuild_prompt_ref_index()

    def _rebuild_prompt_ref_index(self):
        self.prompt_ref_to_idx = {prompt_ref: idx for idx, prompt_ref in enumerate(self.prompt_refs)}

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return (
            idx,
            self.datasources[idx],
            self.prompts[idx],
            self.labels[idx],
            self.knn_pseudo_labels[idx],
            self.late_phase_prompt_flags[idx],
            self.prompt_refs[idx],
        )

    def collate_fn(self, item_list):
        indices = []
        datasources = []
        prompts = []
        labels = []
        knn_pseudo_labels = []
        late_phase_prompt_flags = []
        prompt_refs = []
        for idx, datasource, prompt, label, knn_pl, late_phase_prompt, prompt_ref in item_list:
            indices.append(idx)
            datasources.append(datasource)
            prompts.append(prompt)
            labels.append(label)
            knn_pseudo_labels.append(knn_pl)
            late_phase_prompt_flags.append(late_phase_prompt)
            prompt_refs.append(prompt_ref)
        return indices, datasources, prompts, labels, knn_pseudo_labels, late_phase_prompt_flags, prompt_refs

    def get_sample_by_prompt_ref(self, prompt_ref):
        idx = self.prompt_ref_to_idx[prompt_ref]
        return {
            "idx": idx,
            "prompt_ref": prompt_ref,
            "datasource": self.datasources[idx],
            "prompt": self.prompts[idx],
            "label": self.labels[idx],
            "knn_pseudo_label": self.knn_pseudo_labels[idx],
            "late_phase_prompt": self.late_phase_prompt_flags[idx],
            "raw_record": self.raw_records[idx],
        }

    def iter_prompt_samples(self):
        for prompt_ref in self.prompt_refs:
            yield self.get_sample_by_prompt_ref(prompt_ref)
