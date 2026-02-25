"""
TDC Dataset Loader for GRPO.

Converts TDC CSV files to OpenAI message format with CoT instructions.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional


def levenshtein_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


class TDCDatasetLoader:
    """Loader for TDC molecular property prediction datasets."""

    def __init__(
        self,
        prompts_path: str = "data/tdc/metadata/prompts.json",
        cot_instruction_path: str = "data/tdc/metadata/cot_instruction.txt",
        cot_instruction: Optional[str] = None,
        model_type: Optional[str] = None,
    ):
        """
        Initialize TDC dataset loader.

        Args:
            prompts_path: Path to tdc_prompts.json
            cot_instruction_path: Path to CoT instruction text file
            cot_instruction: CoT instruction to append (overrides file if provided)
            model_type: Model type (e.g. 'gpt-oss') for dynamic prompt customization
        """
        self.prompts = self._load_prompts(prompts_path)

        # Load CoT instruction from file or use provided string
        if cot_instruction:
            self.cot_instruction = cot_instruction
        else:
            self.cot_instruction = self._load_cot_instruction(cot_instruction_path)

        # Apply model-specific string replacements to the CoT instruction
        if model_type and ("gpt_oss" in model_type.lower() or "gpt-oss" in model_type.lower()):
            self.cot_instruction = self.cot_instruction.replace(
                "Please think step by step and use tools when necessary (**Don't use the same tool more than once**).",
                "Please think step by step and use tools when helpful."
            )

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

    def _fuzzy_match_prompt_key(self, task_name: str, max_distance: int = 2) -> Optional[str]:
        """
        Find prompt key using fuzzy matching (case-insensitive, allows up to max_distance edits).
        
        Args:
            task_name: Task name to match
            max_distance: Maximum Levenshtein distance allowed
            
        Returns:
            Matched prompt key or None
        """
        exact_match = next((k for k in self.prompts if k == task_name), None)
        if exact_match:
            return exact_match
        
        # Case-insensitive exact match
        case_insensitive_match = next(
            (k for k in self.prompts if k.lower() == task_name.lower()), 
            None
        )
        if case_insensitive_match:
            return case_insensitive_match
        
        # Fuzzy match with edit distance
        best_match = None
        best_distance = max_distance + 1
        
        for key in self.prompts:
            dist = levenshtein_distance(key.lower(), task_name.lower())
            if dist <= max_distance and dist < best_distance:
                best_match = key
                best_distance = dist
        
        return best_match

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
        # Load CSV first to check schema
        df = pd.read_csv(csv_path)
        
        # Handle Tox21 special case: use task_label column to find prompt
        if task_name == "Tox21" and "task_label" in df.columns:
            return self._load_tox21_csv(csv_path, df)
        
        # Try fuzzy matching for prompt key
        prompt_key = self._fuzzy_match_prompt_key(task_name)
        if not prompt_key:
            raise ValueError(f"No prompt template found for task: {task_name}")
        
        if prompt_key != task_name:
            print(f"  ℹ️  Matched '{task_name}' → '{prompt_key}'")
        
        prompt_template = self.prompts[prompt_key]
        
        # Detect molecule column (Drug, Antibody, etc.)
        mol_column = self._detect_molecule_column(df)
        if not mol_column:
            raise ValueError(f"No molecule column found in CSV (tried: Drug, Antibody, SMILES)")

        # Convert each row
        records = []
        for _, row in df.iterrows():
            smiles = row[mol_column]
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
    
    def _detect_molecule_column(self, df: pd.DataFrame) -> Optional[str]:
        """Detect which column contains the molecule (SMILES/sequence)."""
        for col in ["Drug", "Antibody", "SMILES", "Protein", "Peptide"]:
            if col in df.columns:
                return col
        return None
    
    def _load_tox21_csv(self, csv_path: str, df: pd.DataFrame) -> List[Dict]:
        """
        Load Tox21 CSV with task_label-specific prompts.
        
        Tox21 has subtasks (NR-AR, NR-ER, etc.) indicated by task_label column.
        Each subtask maps to a prompt like Tox21_NR_AR, Tox21_NR_ER, etc.
        """
        records = []
        
        # Get unique task labels
        task_labels = df["task_label"].unique()
        print(f"  ℹ️  Tox21 has {len(task_labels)} subtasks: {sorted(task_labels)}")
        
        for _, row in df.iterrows():
            task_label = row["task_label"]
            
            # Convert task_label (e.g., "NR-AR") to prompt key (e.g., "Tox21_NR_AR")
            # Replace hyphens with underscores
            prompt_key = f"Tox21_{task_label.replace('-', '_')}"
            
            if prompt_key not in self.prompts:
                raise ValueError(f"No prompt found for Tox21 subtask: {prompt_key}")
            
            prompt_template = self.prompts[prompt_key]
            smiles = row["Drug"]
            label = int(row["Y"])
            
            # Build user content
            user_content = self._build_user_content(prompt_template, smiles)
            
            messages = [{"role": "user", "content": user_content}]
            answer = f"({chr(65 + label)})"
            
            records.append({
                "messages": messages,
                "answer": answer,
                "smiles": smiles,
                "label": label,
                "task": f"Tox21_{task_label.replace('-', '_')}",  # Store specific subtask
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
