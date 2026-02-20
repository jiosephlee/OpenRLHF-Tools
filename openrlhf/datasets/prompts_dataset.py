import json
import random
from collections import defaultdict

from torch.utils.data import Dataset
from tqdm import tqdm


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


def preprocess_data(data, input_template=None, input_key="input", label_key=None, apply_chat_template=None, tools_map=None) -> str:
    if apply_chat_template:
        chat = data[input_key]
        if isinstance(chat, str):
            chat = [{"role": "user", "content": chat}]
        kwargs = dict(tokenize=False, add_generation_prompt=True)
        if tools_map is not None:
            # Per-task tool lookup: data["task"] → tool list, fallback to __default__
            task = data.get("task", "__default__")
            kwargs["tools"] = tools_map.get(task, tools_map.get("__default__", []))
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
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.tokenizer = tokenizer

        # chat_template
        self.input_template = input_template
        input_key = getattr(self.strategy.args, "input_key", None)
        label_key = getattr(self.strategy.args, "label_key", None)
        apply_chat_template = getattr(self.strategy.args, "apply_chat_template", False)

        if apply_chat_template:
            apply_chat_template = self.tokenizer.apply_chat_template

        # Load per-task tool schemas: {task_name: [tool_schemas], "__default__": [...]}
        tools_map = None
        tdc_tools_path = getattr(self.strategy.args, "tdc_tools", None)
        if tdc_tools_path:
            with open(tdc_tools_path) as f:
                tools_map = json.load(f)

        self.prompts = []
        self.labels = []
        self.datasources = []
        for data in tqdm(dataset, desc="Preprocessing data", disable=not self.strategy.is_rank_0()):
            prompt, label = preprocess_data(data, input_template, input_key, label_key, apply_chat_template, tools_map=tools_map)
            self.prompts.append(prompt)
            self.labels.append(label)
            self.datasources.append(data.get("datasource", "default"))

        if getattr(self.strategy.args, "curriculum_balanced", False):
            self._apply_curriculum_balancing(getattr(self.strategy.args, "seed", 42))

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

    def __len__(self):
        length = len(self.prompts)
        return length

    def __getitem__(self, idx):
        return idx, self.datasources[idx], self.prompts[idx], self.labels[idx]

    def collate_fn(self, item_list):
        indices = []
        datasources = []
        prompts = []
        labels = []
        for idx, datasource, prompt, label in item_list:
            indices.append(idx)
            datasources.append(datasource)
            prompts.append(prompt)
            labels.append(label)

        return indices, datasources, prompts, labels
