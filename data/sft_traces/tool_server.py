#!/usr/bin/env python3
"""
Live tool server for TDC-RL-16 RL rollouts.

Provides two callable tools that compute molecular features from SMILES:
  1. get_molecular_descriptors(smiles, task) -> RDKit features with semantic descriptions
  2. get_similar_molecules(smiles, task, k=3) -> KNN neighbors with labels + aggregate stats

Usage:
    server = ToolServer(base_dir="/path/to/sft_traces")
    server.load_task("AMES")

    # During RL rollout, when model emits a tool_call:
    result = server.call_tool("get_molecular_descriptors", {"smiles": "CCO"}, task="AMES")
    result = server.call_tool("get_similar_molecules", {"smiles": "CCO", "k": 3}, task="AMES")

    # Get tool definitions for a task (pass to model as `tools` parameter):
    tools = server.get_tools("AMES")

    # Get system prompt + user prompt for a SMILES:
    system, user = server.get_prompts("AMES", "CCO")

    # Reward: extract model's (A)/(B) answer and compare to ground truth
    reward = server.compute_reward(model_output_text, true_label=1)
"""

import os
import sys
import json
import re
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.impute import SimpleImputer
import joblib


TASKS = [
    'AMES', 'BBB_Martins', 'Bioavailability_Ma',
    'CYP2C9_Substrate_CarbonMangels', 'CYP2D6_Substrate_CarbonMangels',
    'CYP3A4_Substrate_CarbonMangels', 'Carcinogens_Lagunin', 'ClinTox',
    'DILI', 'HIA_Hou', 'PAMPA_NCATS', 'Pgp_Broccatelli',
    'SARSCoV2_3CLPro_Diamond', 'SARSCoV2_Vitro_Touret', 'Skin_Reaction', 'hERG',
]

KNN_FEATURE_NAMES = ['KNN_mean_label', 'KNN_min_dist', 'KNN_mean_dist']

# Regex to extract (A) or (B) from model output
RE_ANSWER = re.compile(r'(?:Answer:\s*)?(\([AB]\))')


class ToolServer:
    """Serves live molecular tools for RL rollouts."""

    def __init__(self, base_dir, desc_dir=None):
        """
        Args:
            base_dir: Path to sft_traces/ directory (contains rf_models/, etc.)
            desc_dir: Path to directory with *_descriptions.json files.
                      Defaults to base_dir itself, then parent.
        """
        self.base_dir = os.path.abspath(base_dir)
        self.models_dir = os.path.join(self.base_dir, 'rf_models')
        parent_dir = os.path.dirname(self.base_dir)
        self.task_formats_dir = os.path.join(
            parent_dir, 'tdc_eval', 'base_no_tools', 'task_formats')
        self.splits_dir = os.path.join(
            parent_dir, 'tdc_eval', 'reference_splits')

        if desc_dir is None:
            if os.path.exists(os.path.join(self.base_dir, 'rdkit_descriptor_descriptions.json')):
                desc_dir = self.base_dir
            else:
                desc_dir = parent_dir
        self.desc_dir = os.path.abspath(desc_dir)

        # RDKit descriptor calculator (shared across tasks)
        self.rdkit_desc_names = [x[0] for x in Descriptors._descList]
        self.rdkit_calc = MoleculeDescriptors.MolecularDescriptorCalculator(
            self.rdkit_desc_names)

        # Load semantic description maps
        self.rdkit_descs, self.knn_descs = self._load_descriptions()

        # Per-task state (lazy-loaded)
        self._task_state = {}  # task -> {model_data, train_fps, train_smiles, train_labels, ...}
        self._task_formats = {}

    def _load_descriptions(self):
        rdkit_path = os.path.join(self.desc_dir, 'rdkit_descriptor_descriptions.json')
        with open(rdkit_path) as f:
            rdkit_descs = json.load(f)

        knn_descs = {
            'KNN_mean_label': 'Mean label of the k nearest training neighbors (0 = all negative, 1 = all positive).',
            'KNN_min_dist': 'Tanimoto distance to the closest training neighbor (0 = identical fingerprint).',
            'KNN_mean_dist': 'Mean Tanimoto distance across all k nearest training neighbors.',
        }
        return rdkit_descs, knn_descs

    def load_task(self, task):
        """Pre-load a task's RF model, training data, and fingerprints."""
        if task in self._task_state:
            return

        print(f"Loading task: {task}")

        # Load RF model + metadata
        model_path = os.path.join(self.models_dir, f'{task}.joblib')
        model_data = joblib.load(model_path)

        # Load training data (needed for KNN)
        train_df = pd.read_csv(os.path.join(self.splits_dir, f'{task}_train.csv'))
        train_smiles = list(train_df['smiles'])
        train_labels = np.array(train_df.iloc[:, 1])

        # Compute training fingerprints for KNN
        train_fps = self._compute_morgan_fps(train_smiles)

        # Load task format
        fmt_path = os.path.join(self.task_formats_dir, f'{task}.json')
        with open(fmt_path) as f:
            task_format = json.load(f)

        self._task_state[task] = {
            'model_data': model_data,
            'train_fps': train_fps,
            'train_smiles': train_smiles,
            'train_labels': train_labels,
            'feature_names': model_data['feature_names'],
            'imputer': model_data['imputer'],
            'keep_mask': model_data['keep_mask'],
        }
        self._task_formats[task] = task_format

    def _compute_morgan_fps(self, smiles_list):
        fps = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi) if not pd.isna(smi) else None
            if mol is not None:
                fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
            else:
                fps.append(None)
        return fps

    def _compute_rdkit(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return {name: np.nan for name in self.rdkit_desc_names}
        vals = self.rdkit_calc.CalcDescriptors(mol)
        return dict(zip(self.rdkit_desc_names, vals))

    def _compute_knn(self, query_fp, task, k=3):
        state = self._task_state[task]
        train_fps = state['train_fps']
        train_smiles = state['train_smiles']
        train_labels = state['train_labels']

        if query_fp is None:
            return [], {
                'KNN_mean_label': float(np.mean(train_labels)),
                'KNN_min_dist': 1.0,
                'KNN_mean_dist': 1.0,
            }

        sims = np.array([
            DataStructs.TanimotoSimilarity(query_fp, tfp) if tfp is not None else 0.0
            for tfp in train_fps
        ])
        k_actual = min(k, len(sims))
        top_k_idx = np.argsort(sims)[-k_actual:][::-1]

        neighbors = []
        for idx in top_k_idx:
            neighbors.append({
                'smiles': train_smiles[idx],
                'similarity': float(sims[idx]),
                'distance': float(1.0 - sims[idx]),
                'label': 'positive' if int(train_labels[idx]) == 1 else 'negative',
            })

        dists = [n['distance'] for n in neighbors]
        labels_numeric = [1 if n['label'] == 'positive' else 0 for n in neighbors]

        stats = {
            'KNN_mean_label': float(np.mean(labels_numeric)),
            'KNN_min_dist': float(min(dists)),
            'KNN_mean_dist': float(np.mean(dists)),
        }
        return neighbors, stats

    # =========================================================================
    # Public API
    # =========================================================================

    def get_tools(self, task):
        """Return OpenAI-format tool definitions for a task."""
        descriptor_description = (
            "Compute molecular descriptors for a given SMILES string. Returns a dictionary of "
            "numerical features derived from the molecule's structure. Each feature includes its "
            "value and a semantic description explaining what it measures."
        )
        knn_description = (
            "Find the k most similar molecules in the training database using Tanimoto similarity "
            "on Morgan fingerprints (radius=2, 2048 bits). Returns each neighbor's SMILES, "
            "similarity, distance, and known label, plus aggregate statistics."
        )
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_molecular_descriptors",
                    "description": descriptor_description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "smiles": {"type": "string", "description": "SMILES string of the molecule."}
                        },
                        "required": ["smiles"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_similar_molecules",
                    "description": knn_description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "smiles": {"type": "string", "description": "SMILES string of the query molecule."},
                            "k": {"type": "integer", "description": "Number of neighbors.", "default": 3}
                        },
                        "required": ["smiles"]
                    }
                }
            }
        ]

    def get_prompts(self, task, smiles):
        """Return (system_prompt, user_prompt) for a task + SMILES."""
        fmt = self._task_formats[task]
        system = fmt['system_prompt']
        user = fmt['template'].replace('{smiles}', smiles)
        if "Please think step by step" in user:
            user = user.split("Please think step by step")[0].strip()
        return system, user

    def call_tool(self, tool_name, arguments, task):
        """
        Execute a tool call and return the result as a JSON string.

        Args:
            tool_name: "get_molecular_descriptors" or "get_similar_molecules"
            arguments: dict with tool arguments (parsed from model's function call)
            task: task name (e.g., "AMES")

        Returns:
            JSON string (same format as the SFT training data tool responses)
        """
        self.load_task(task)
        smiles = arguments.get('smiles', '')

        if tool_name == 'get_molecular_descriptors':
            return self._tool_descriptors(smiles, task)
        elif tool_name == 'get_similar_molecules':
            k = arguments.get('k', 3)
            return self._tool_knn(smiles, task, k)
        else:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def _tool_descriptors(self, smiles, task):
        """Compute RDKit descriptors for a SMILES, return with semantic descriptions."""
        state = self._task_state[task]

        # Compute raw RDKit features
        rdkit_feats = self._compute_rdkit(smiles)

        # Filter to features the RF model actually uses
        # (after zero-variance removal during training)
        feature_names = state['feature_names']
        # Exclude KNN features — those come from get_similar_molecules
        non_knn_names = [f for f in feature_names if f not in KNN_FEATURE_NAMES]

        # Build result with semantic descriptions
        desc_map = dict(self.rdkit_descs)

        result = {"smiles": smiles, "descriptors": {}}
        for fname in non_knn_names:
            val = rdkit_feats.get(fname, np.nan)
            if pd.isna(val) or not np.isfinite(val):
                val = None  # will be imputed by model or shown as null
            else:
                val = float(val)

            entry = {"value": val}
            if fname in desc_map:
                entry["description"] = desc_map[fname]
            result["descriptors"][fname] = entry

        return json.dumps(result)

    def _tool_knn(self, smiles, task, k=3):
        """Find k nearest neighbors and return with descriptions."""
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        query_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None

        neighbors, stats = self._compute_knn(query_fp, task, k)

        result = {
            "query_smiles": smiles,
            "k": k,
            "neighbors": neighbors,
            "aggregate_stats": {}
        }
        for stat_name, stat_val in stats.items():
            entry = {"value": stat_val}
            if stat_name in self.knn_descs:
                entry["description"] = self.knn_descs[stat_name]
            result["aggregate_stats"][stat_name] = entry

        return json.dumps(result)

    def get_val_examples(self, task):
        """Load validation examples for a task. Returns list of (smiles, label)."""
        val_df = pd.read_csv(os.path.join(self.splits_dir, f'{task}_valid.csv'))
        return list(zip(val_df['smiles'], val_df.iloc[:, 1]))

    def get_train_examples(self, task):
        """Load training examples. Returns list of (smiles, label)."""
        train_df = pd.read_csv(os.path.join(self.splits_dir, f'{task}_train.csv'))
        return list(zip(train_df['smiles'], train_df.iloc[:, 1]))

    @staticmethod
    def compute_reward(model_output, true_label):
        """
        Extract (A)/(B) from model output and return binary reward.

        Args:
            model_output: The model's final assistant message text
            true_label: 0 or 1 (ground truth)

        Returns:
            1.0 if correct, 0.0 if wrong, -0.1 if no valid answer extracted
        """
        matches = RE_ANSWER.findall(model_output)
        if not matches:
            return -0.1  # penalty for no parseable answer

        predicted = matches[-1]  # take last match (after reasoning)
        expected = "(B)" if true_label == 1 else "(A)"
        return 1.0 if predicted == expected else 0.0


# =============================================================================
# Convenience: run a quick smoke test
# =============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Test tool server')
    parser.add_argument('--base-dir', default=os.path.dirname(os.path.abspath(__file__)),
                        help='Path to sft_traces/ directory')
    parser.add_argument('--task', default='AMES')
    parser.add_argument('--smiles', default='c1ccccc1')  # benzene
    args = parser.parse_args()

    server = ToolServer(args.base_dir)
    server.load_task(args.task)

    print(f"\n=== Tool Server Smoke Test: {args.task} ===")
    print(f"SMILES: {args.smiles}")

    # Test get_molecular_descriptors
    result = server.call_tool("get_molecular_descriptors", {"smiles": args.smiles}, args.task)
    parsed = json.loads(result)
    n_feats = len(parsed['descriptors'])
    sample_feats = list(parsed['descriptors'].items())[:3]
    print(f"\nget_molecular_descriptors: {n_feats} features")
    for fname, fdata in sample_feats:
        print(f"  {fname}: value={fdata['value']}, desc={fdata.get('description', 'N/A')[:60]}...")

    # Test get_similar_molecules
    result = server.call_tool("get_similar_molecules", {"smiles": args.smiles, "k": 3}, args.task)
    parsed = json.loads(result)
    print(f"\nget_similar_molecules: {len(parsed['neighbors'])} neighbors")
    for nbr in parsed['neighbors']:
        print(f"  {nbr['smiles'][:40]}... sim={nbr['similarity']:.3f} label={nbr['label']}")
    for stat, data in parsed['aggregate_stats'].items():
        print(f"  {stat}: {data['value']:.3f}")

    # Test prompts
    system, user = server.get_prompts(args.task, args.smiles)
    print(f"\nSystem prompt: {system[:80]}...")
    print(f"User prompt: {user[:80]}...")

    # Test tools
    tools = server.get_tools(args.task)
    print(f"\nTool definitions: {len(tools)} tools")
    for t in tools:
        print(f"  {t['function']['name']}")

    # Test reward
    print(f"\nReward tests:")
    print(f"  'Answer: (A)' vs label=0 -> {server.compute_reward('Answer: (A)', 0)}")
    print(f"  'Answer: (B)' vs label=0 -> {server.compute_reward('Answer: (B)', 0)}")
    print(f"  'no answer' vs label=1   -> {server.compute_reward('no answer here', 1)}")

    # Show val examples
    val = server.get_val_examples(args.task)
    print(f"\nValidation examples: {len(val)} total")
    print(f"  Label distribution: {sum(l for _, l in val)} positive, {sum(1-l for _, l in val)} negative")

    print("\n=== Smoke test passed ===")
