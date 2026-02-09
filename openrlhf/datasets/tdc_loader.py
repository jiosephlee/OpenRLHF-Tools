"""
TDC Dataset Loader for GRPO.

Converts TDC CSV files to OpenAI message format with CoT instructions.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


class TDCDatasetLoader:
    """Loader for TDC molecular property prediction datasets."""

    def __init__(
        self,
        prompts_path: str = "data/tdc/metadata/prompts.json",
        cot_instruction_path: str = "data/tdc/metadata/cot_instruction.txt",
        cot_instruction: Optional[str] = None,
    ):
        """
        Initialize TDC dataset loader.

        Args:
            prompts_path: Path to tdc_prompts.json
            cot_instruction_path: Path to CoT instruction text file
            cot_instruction: CoT instruction to append (overrides file if provided)
        """
        self.prompts = self._load_prompts(prompts_path)

        # Load CoT instruction from file or use provided string
        if cot_instruction:
            self.cot_instruction = cot_instruction
        else:
            self.cot_instruction = self._load_cot_instruction(cot_instruction_path)

    def _load_prompts(self, prompts_path: str) -> Dict[str, str]:
        """Load TDC prompt templates."""
        with open(prompts_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_cot_instruction(self, cot_instruction_path: str) -> str:
        """Load CoT instruction from text file."""
        with open(cot_instruction_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def _strip_answer_suffix(self, prompt: str) -> str:
        """
        Strip 'Answer:' suffix from TDC prompt templates.

        TDC prompts end with "...Answer:" but for GRPO we want the model
        to generate this part, so we remove it.
        """
        if prompt.strip().endswith("Answer:"):
            prompt = prompt.rsplit("Answer:", 1)[0].strip()
        return prompt

    def _build_user_content(self, prompt_template: str, smiles: str) -> str:
        """
        Build user message content with CoT instructions.

        Args:
            prompt_template: Base prompt from tdc_prompts.json
            smiles: SMILES string

        Returns:
            User content with SMILES filled in and CoT instructions appended
        """
        # Strip "Answer:" suffix
        prompt_stripped = self._strip_answer_suffix(prompt_template)

        # Replace SMILES placeholder
        user_content = prompt_stripped.replace("{Drug SMILES}", smiles)

        # Append CoT instructions
        user_content += self.cot_instruction

        return user_content

    def load_csv_to_openai_format(
        self,
        csv_path: str,
        task_name: str,
    ) -> List[Dict]:
        """
        Load a single CSV file and convert to OpenAI message format.

        Args:
            csv_path: Path to train.csv, test.csv, or val.csv
            task_name: Task name (e.g., "AMES")

        Returns:
            List of records with OpenAI message format
        """
        # Get prompt template
        if task_name not in self.prompts:
            raise ValueError(f"No prompt template found for task: {task_name}")

        prompt_template = self.prompts[task_name]

        # Load CSV
        df = pd.read_csv(csv_path)

        # Convert each row
        records = []
        for _, row in df.iterrows():
            smiles = row["Drug"]
            label = int(row["Y"])

            # Build user content with CoT instructions
            user_content = self._build_user_content(prompt_template, smiles)

            # Create OpenAI message format
            messages = [
                {"role": "user", "content": user_content}
            ]

            # Convert label: 0 → (A), 1 → (B)
            answer = f"({chr(65 + label)})"

            records.append({
                "messages": messages,
                "answer": answer,
                "smiles": smiles,
                "label": label,
                "task": task_name,
            })

        return records

    def convert_task(
        self,
        task_name: str,
        raw_dir: str,
        output_dir: str,
        splits: List[str] = ["train", "val", "test"],
    ):
        """
        Convert all splits of a single task to JSONL.

        Args:
            task_name: Task name (e.g., "AMES")
            raw_dir: Directory containing raw CSV files
            output_dir: Output directory for JSONL files
            splits: List of splits to convert
        """
        task_dir = Path(raw_dir) / task_name

        if not task_dir.exists():
            raise ValueError(f"Task directory not found: {task_dir}")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for split in splits:
            csv_path = task_dir / f"{split}.csv"
            if not csv_path.exists():
                print(f"  ⚠️  {split}.csv not found, skipping")
                continue

            # Convert to OpenAI format
            records = self.load_csv_to_openai_format(str(csv_path), task_name)

            # Write JSONL
            output_file = output_path / f"{task_name}_{split}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for record in records:
                    f.write(json.dumps(record, ensure_ascii=False) + '\n')

            print(f"  ✓ Wrote {len(records)} records to {output_file}")

    def convert_all_tasks(
        self,
        raw_dir: str,
        output_dir: str,
        splits: List[str] = ["train", "val", "test"],
    ):
        """
        Convert all tasks in raw_dir to JSONL.

        Args:
            raw_dir: Directory containing TDC task subdirectories
            output_dir: Output directory for JSONL files
            splits: List of splits to convert
        """
        raw_path = Path(raw_dir)

        for task_dir in sorted(raw_path.iterdir()):
            if task_dir.is_dir():
                try:
                    print(f"\nProcessing {task_dir.name}...")
                    self.convert_task(task_dir.name, raw_dir, output_dir, splits)
                except Exception as e:
                    print(f"  ✗ Error processing {task_dir.name}: {e}")
