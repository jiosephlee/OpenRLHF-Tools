"""Build joint multi-endpoint TDC datasets from overlapping molecules.

Finds molecules shared across related tasks and creates merged datasets
where each sample asks the model to predict multiple endpoints at once.

Usage:
    python data/tdc/build_joint_datasets.py [--min-overlap 50] [--output-dir data/tdc/joint_format]
"""

import os
import json
import argparse
import pandas as pd
from collections import OrderedDict
from itertools import combinations
from rdkit import Chem


RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
PROMPTS_PATH = os.path.join(os.path.dirname(__file__), "metadata", "prompts.json")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "joint_format")

# ── Task group definitions ──────────────────────────────────────────────

JOINT_DATASETS = OrderedDict({
    # CYP inhibition panel — largest overlap
    "CYP5_Inhibition": {
        "tasks": ["CYP1A2_Veith", "CYP2C19_Veith", "CYP2C9_Veith", "CYP2D6_Veith", "CYP3A4_Veith"],
        "category": "ADME / CYP450 Metabolism",
        "context": (
            "Cytochrome P450 (CYP) enzymes are responsible for metabolizing the majority of "
            "clinically used drugs. Inhibition of these enzymes by a drug can cause dangerous "
            "drug-drug interactions by raising plasma levels of co-administered medications. "
            "The five major CYP isoforms — CYP1A2, CYP2C9, CYP2C19, CYP2D6, and CYP3A4 — "
            "together account for ~90%% of drug metabolism. A drug's CYP inhibition profile "
            "across these isoforms is a critical factor in clinical safety assessment."
        ),
        "endpoint_descriptions": {
            "CYP1A2_Veith": "CYP1A2 inhibition (metabolizes caffeine, theophylline, PAHs)",
            "CYP2C19_Veith": "CYP2C19 inhibition (metabolizes clopidogrel, omeprazole)",
            "CYP2C9_Veith": "CYP2C9 inhibition (metabolizes warfarin, NSAIDs)",
            "CYP2D6_Veith": "CYP2D6 inhibition (metabolizes antipsychotics, beta-blockers)",
            "CYP3A4_Veith": "CYP3A4 inhibition (metabolizes ~50%% of all drugs)",
        },
        "labels": {0: "non-inhibitor", 1: "inhibitor"},
    },

    "CYP3_Inhibition": {
        "tasks": ["CYP1A2_Veith", "CYP2C19_Veith", "CYP3A4_Veith"],
        "category": "ADME / CYP450 Metabolism",
        "context": (
            "Cytochrome P450 (CYP) enzymes metabolize the majority of drugs. Inhibition of "
            "CYP1A2, CYP2C19, or CYP3A4 can cause drug-drug interactions. CYP3A4 alone "
            "metabolizes ~50%% of all drugs, making its inhibition profile especially important."
        ),
        "endpoint_descriptions": {
            "CYP1A2_Veith": "CYP1A2 inhibition",
            "CYP2C19_Veith": "CYP2C19 inhibition",
            "CYP3A4_Veith": "CYP3A4 inhibition",
        },
        "labels": {0: "non-inhibitor", 1: "inhibitor"},
    },

    "CYP3_Substrate": {
        "tasks": ["CYP2C9_Substrate_CarbonMangels", "CYP2D6_Substrate_CarbonMangels", "CYP3A4_Substrate_CarbonMangels"],
        "category": "ADME / CYP450 Metabolism",
        "context": (
            "CYP substrate recognition determines which enzymes metabolize a drug, affecting "
            "its clearance rate, half-life, and susceptibility to drug-drug interactions. "
            "CYP2C9, CYP2D6, and CYP3A4 are three of the most important metabolizing enzymes."
        ),
        "endpoint_descriptions": {
            "CYP2C9_Substrate_CarbonMangels": "CYP2C9 substrate",
            "CYP2D6_Substrate_CarbonMangels": "CYP2D6 substrate",
            "CYP3A4_Substrate_CarbonMangels": "CYP3A4 substrate",
        },
        "labels": {0: "non-substrate", 1: "substrate"},
    },

    "Tox_AMES_Tox21": {
        "tasks": ["AMES", "Tox21"],
        "category": "Toxicity",
        "context": (
            "Genotoxicity and cellular stress are two major axes of drug toxicity assessment. "
            "The Ames test measures mutagenicity via bacterial reverse mutation. Tox21 screens "
            "compounds across nuclear receptor (NR) and stress response (SR) pathways using "
            "high-throughput assays. A compound's combined Ames and Tox21 profile reveals "
            "whether it causes DNA damage and/or disrupts critical cellular signaling."
        ),
        "endpoint_descriptions": {
            "AMES": "Ames mutagenicity",
            "Tox21": "Tox21 activity (specify subtask in label)",
        },
        "labels": {0: "inactive/negative", 1: "active/positive"},
    },

    "ADME_Oral": {
        "tasks": ["BBB_Martins", "Bioavailability_Ma"],
        "category": "ADME / Absorption",
        "context": (
            "Oral drug absorption depends on two key barriers: intestinal absorption into "
            "systemic circulation (bioavailability) and penetration across the blood-brain "
            "barrier (BBB) for CNS targets. A drug's combined BBB and bioavailability profile "
            "determines its suitability for oral CNS drug development."
        ),
        "endpoint_descriptions": {
            "BBB_Martins": "Blood-brain barrier penetration",
            "Bioavailability_Ma": "Oral bioavailability (F >= 20%%)",
        },
        "labels": {0: "negative/low", 1: "positive/high"},
    },
})


def canonicalize(smiles: str) -> str | None:
    """Canonicalize a SMILES string. Returns None on failure."""
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def load_task_split(task: str, split: str) -> pd.DataFrame:
    """Load a single task/split CSV, canonicalize SMILES, return DataFrame."""
    path = os.path.join(RAW_DIR, task, f"{split}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["canon_smiles"] = df["Drug"].apply(canonicalize)
    df = df.dropna(subset=["canon_smiles"])
    return df


def build_joint_dataset(name: str, config: dict, output_dir: str = OUTPUT_DIR) -> dict:
    """Build a joint dataset for the given config. Returns stats dict."""
    tasks = config["tasks"]
    stats = {"name": name, "tasks": tasks, "splits": {}}

    for split in ["train", "val", "test"]:
        # Load all tasks for this split
        task_dfs = {}
        for task in tasks:
            df = load_task_split(task, split)
            if df.empty:
                continue

            # For Tox21, pivot subtasks into separate columns
            if task == "Tox21" and "task_label" in df.columns:
                subtask_dfs = {}
                for subtask, sub_df in df.groupby("task_label"):
                    col_name = f"Tox21_{subtask}"
                    sub_df = sub_df.drop_duplicates(subset="canon_smiles")
                    subtask_dfs[col_name] = sub_df.set_index("canon_smiles")["Y"]
                # Find molecules that appear in at least one subtask
                all_smiles = set()
                for s in subtask_dfs.values():
                    all_smiles.update(s.index)
                tox21_merged = pd.DataFrame(index=list(all_smiles))
                for col_name, series in subtask_dfs.items():
                    tox21_merged[col_name] = series
                task_dfs["Tox21"] = tox21_merged
            else:
                df_dedup = df.drop_duplicates(subset="canon_smiles")
                task_dfs[task] = df_dedup.set_index("canon_smiles")[["Y"]].rename(columns={"Y": task})

        if not task_dfs:
            continue

        # Find overlapping molecules
        all_indices = [set(df.index) for df in task_dfs.values()]
        overlap_smiles = set.intersection(*all_indices)

        if not overlap_smiles:
            stats["splits"][split] = {"count": 0}
            continue

        # Merge into single DataFrame
        merged = pd.DataFrame(index=list(overlap_smiles))
        endpoint_cols = []
        for task, df in task_dfs.items():
            if task == "Tox21":
                for col in df.columns:
                    merged[col] = df.loc[merged.index, col]
                    endpoint_cols.append(col)
            else:
                merged[task] = df.loc[merged.index, task]
                endpoint_cols.append(task)

        merged = merged.dropna(subset=endpoint_cols, how="all")
        merged.index.name = "canon_smiles"

        stats["splits"][split] = {"count": len(merged), "endpoints": endpoint_cols}

        # Generate JSONL
        records = _build_records(merged, endpoint_cols, config, name)

        # Write output
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"{name}_{split}.jsonl")
        with open(out_path, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")

        print(f"  {split}: {len(records)} records → {out_path}")

    return stats


def _build_records(merged: pd.DataFrame, endpoint_cols: list, config: dict, dataset_name: str) -> list:
    """Build JSONL records with multi-endpoint prompt."""
    records = []
    label_map = config["labels"]
    endpoint_descs = config.get("endpoint_descriptions", {})

    for smiles, row in merged.iterrows():
        # Build the endpoint labels dict (skip NaN for Tox21 subtasks)
        labels = {}
        for col in endpoint_cols:
            val = row.get(col)
            if pd.notna(val):
                labels[col] = int(val)

        if not labels:
            continue

        # Format the question
        prompt = _format_prompt(smiles, labels, config, endpoint_descs)

        # Format the expected answer
        answer = _format_answer(labels, label_map)

        records.append({
            "messages": [{"role": "user", "content": prompt}],
            "answer": answer,
            "smiles": smiles,
            "labels": labels,
            "task": dataset_name,
            "endpoint_tasks": list(labels.keys()),
        })

    return records


def _format_prompt(smiles: str, labels: dict, config: dict, endpoint_descs: dict) -> str:
    """Format the multi-endpoint prediction prompt."""
    context = config["context"]
    category = config["category"]
    label_map = config["labels"]

    # Build endpoint list
    endpoint_lines = []
    for i, (task_name, _) in enumerate(labels.items(), 1):
        desc = endpoint_descs.get(task_name, task_name)
        neg = label_map[0]
        pos = label_map[1]
        endpoint_lines.append(f"  {i}. {desc}: (A) {neg} or (B) {pos}")

    endpoints_str = "\n".join(endpoint_lines)

    prompt = (
        f"Instructions: Predict the following drug endpoints.\n"
        f"Context: {context}\n"
        f"Drug SMILES: {smiles}\n\n"
        f"Predict each endpoint for this molecule:\n"
        f"{endpoints_str}\n\n"
        f"Please think step by step and use tools when helpful. "
        f"Then provide your predictions in the following format:\n"
        f"Predictions:\n"
    )

    # Show expected format
    for i, (task_name, _) in enumerate(labels.items(), 1):
        short_name = _short_name(task_name)
        prompt += f"  {short_name}: (A) or (B)\n"

    return prompt


def _format_answer(labels: dict, label_map: dict) -> str:
    """Format the expected answer in extractable format."""
    lines = ["Predictions:"]
    for task_name, label in labels.items():
        short_name = _short_name(task_name)
        choice = "(A)" if label == 0 else "(B)"
        desc = label_map[label]
        lines.append(f"  {short_name}: {choice} {desc}")
    return "\n".join(lines)


def _short_name(task_name: str) -> str:
    """Convert task name to short display name."""
    mapping = {
        "CYP1A2_Veith": "CYP1A2",
        "CYP2C19_Veith": "CYP2C19",
        "CYP2C9_Veith": "CYP2C9",
        "CYP2D6_Veith": "CYP2D6",
        "CYP3A4_Veith": "CYP3A4",
        "CYP2C9_Substrate_CarbonMangels": "CYP2C9_Sub",
        "CYP2D6_Substrate_CarbonMangels": "CYP2D6_Sub",
        "CYP3A4_Substrate_CarbonMangels": "CYP3A4_Sub",
        "BBB_Martins": "BBB",
        "Bioavailability_Ma": "Bioavailability",
    }
    # Handle Tox21 subtasks
    if task_name.startswith("Tox21_"):
        return task_name.replace("Tox21_", "Tox21-")
    return mapping.get(task_name, task_name)


def main():
    parser = argparse.ArgumentParser(description="Build joint multi-endpoint TDC datasets")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Specific datasets to build (default: all)")
    args = parser.parse_args()

    output_dir = args.output_dir

    all_stats = []
    targets = args.datasets or list(JOINT_DATASETS.keys())

    for name in targets:
        if name not in JOINT_DATASETS:
            print(f"Unknown dataset: {name}")
            continue
        print(f"\n{'='*60}")
        print(f"Building: {name}")
        print(f"{'='*60}")
        config = JOINT_DATASETS[name]
        stats = build_joint_dataset(name, config, output_dir=output_dir)
        all_stats.append(stats)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for s in all_stats:
        train_n = s["splits"].get("train", {}).get("count", 0)
        val_n = s["splits"].get("val", {}).get("count", 0)
        test_n = s["splits"].get("test", {}).get("count", 0)
        n_endpoints = len(s["splits"].get("train", {}).get("endpoints", s["tasks"]))
        print(f"  {s['name']}: {train_n} train / {val_n} val / {test_n} test  ({n_endpoints} endpoints)")

    # Save metadata
    meta_path = os.path.join(output_dir, "joint_datasets_meta.json")
    with open(meta_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"\nMetadata → {meta_path}")


if __name__ == "__main__":
    main()
