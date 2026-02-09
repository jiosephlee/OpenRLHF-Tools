"""
TDC GRPO Dataset Loader.

Loads TDC datasets in OpenAI message format and applies tokenizer's chat template
with tools at load time for GRPO training.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional

from datasets import Dataset


class TDCGRPODataset:
    """Dataset loader for TDC GRPO training."""

    def __init__(
        self,
        data_path: str,
        tokenizer,
        tool_mode: str = "TaskSpecific",
        task_name: Optional[str] = None,
        tools_all_path: str = "data/tdc/metadata/tools_all.json",
        tools_task_specific_path: str = "data/tdc/metadata/tools_task_specific.json",
    ):
        """
        Initialize TDC GRPO dataset loader.

        Args:
            data_path: Path to JSONL file with OpenAI message format
            tokenizer: Tokenizer with apply_chat_template method
            tool_mode: "TaskSpecific", "All", or "None"
            task_name: Task name (required if tool_mode="TaskSpecific")
            tools_all_path: Path to tools_all.json
            tools_task_specific_path: Path to tools_task_specific.json
        """
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.tool_mode = tool_mode
        self.task_name = task_name
        self.tools_all_path = tools_all_path
        self.tools_task_specific_path = tools_task_specific_path

        # Load tools
        self.tools = self._load_tools()

        # Load data
        self.data = self._load_data()

    def _load_tools(self) -> List[Dict]:
        """Load tool definitions based on tool_mode."""
        if self.tool_mode == "None":
            return []
        elif self.tool_mode == "TaskSpecific":
            if not self.task_name:
                raise ValueError("task_name required when tool_mode='TaskSpecific'")

            # Load task-specific tools from JSON
            with open(self.tools_task_specific_path, 'r') as f:
                task_tools = json.load(f)

            tools = task_tools.get(self.task_name, [])
            if not tools:
                print(f"Warning: No task-specific tools found for {self.task_name}, using all tools")
                with open(self.tools_all_path, 'r') as f:
                    tools = json.load(f)
            return tools
        else:  # "All"
            # Load all tools from JSON
            with open(self.tools_all_path, 'r') as f:
                return json.load(f)

    def _load_data(self) -> List[Dict]:
        """Load JSONL data."""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                record = json.loads(line)
                data.append(record)
        return data

    def _apply_chat_template(self, messages: List[Dict]) -> str:
        """Apply tokenizer's chat template with tools."""
        return self.tokenizer.apply_chat_template(
            messages,
            tools=self.tools if self.tools else None,
            add_generation_prompt=True,
            tokenize=False,
        )

    def to_dataset(self) -> Dataset:
        """
        Convert to HuggingFace Dataset with "question" and "answer" fields.

        Returns:
            Dataset with columns: question (str), answer (str), smiles (str), label (int)
        """
        processed_data = []
        for record in self.data:
            messages = record["messages"]
            answer = record["answer"]
            smiles = record.get("smiles", "")
            label = record.get("label", 0)

            # Apply chat template to get the question
            question = self._apply_chat_template(messages)

            processed_data.append({
                "question": question,
                "answer": answer,
                "smiles": smiles,
                "label": label,
            })

        return Dataset.from_list(processed_data)


def load_tdc_grpo_dataset(
    data_path: str,
    tokenizer,
    tool_mode: str = "TaskSpecific",
    task_name: Optional[str] = None,
    tools_all_path: str = "data/tdc/metadata/tools_all.json",
    tools_task_specific_path: str = "data/tdc/metadata/tools_task_specific.json",
) -> Dataset:
    """
    Convenience function to load TDC GRPO dataset.

    Args:
        data_path: Path to JSONL file
        tokenizer: Tokenizer with apply_chat_template
        tool_mode: "TaskSpecific", "All", or "None"
        task_name: Task name (required if tool_mode="TaskSpecific")
        tools_all_path: Path to tools_all.json
        tools_task_specific_path: Path to tools_task_specific.json

    Returns:
        HuggingFace Dataset with "question" and "answer" fields
    """
    loader = TDCGRPODataset(
        data_path, tokenizer, tool_mode, task_name,
        tools_all_path, tools_task_specific_path
    )
    return loader.to_dataset()
