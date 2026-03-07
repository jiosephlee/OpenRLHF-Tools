import os
import glob
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Paths ---
EVAL_DIR = "/Users/jlee0/Desktop/research/OpenRLHF-Tools/runs/grpo-tdc-s1-16t-v4-ep1-sr3-0303_0544/eval_metrics"


def extract_step(filepath):
    filename = os.path.basename(filepath)
    step_str = filename.replace("eval_step_", "").replace(".json", "")
    return int(step_str)


def main():
    # 1. Get and Sort Eval Files
    files = glob.glob(os.path.join(EVAL_DIR, "eval_step_*.json"))
    if not files:
        print(f"No JSON files found in {EVAL_DIR}")
        return

    sorted_files = sorted(files, key=extract_step)

    # 2. Load Step 0 Baseline Data and Dataset Counts
    step_0_file = os.path.join(EVAL_DIR, "eval_step_0.json")
    if not os.path.exists(step_0_file):
        print(f"Error: Baseline file {step_0_file} not found! Needed for deltas and counts.")
        return

    with open(step_0_file, "r") as f:
        baseline_data = json.load(f)["per_task"]

    # Extract the dataset count for each task
    count_dict = {task: metrics.get("count", 0) for task, metrics in baseline_data.items()}

    # 3. Data Collection Arrays
    steps_static, static_pearson, static_spearman = [], [], []
    steps_rel, rel_pearson, rel_spearman = [], [], []

    for filepath in sorted_files:
        step = extract_step(filepath)
        with open(filepath, "r") as f:
            data = json.load(f)["per_task"]

        task_records = []
        for task, metrics in data.items():
            if task not in baseline_data or task not in count_dict:
                continue

            acc_t = metrics.get("accuracy", 0)
            f1_t = metrics.get("macro_f1", 0)
            acc_0 = baseline_data[task].get("accuracy", 0)
            f1_0 = baseline_data[task].get("macro_f1", 0)
            count = count_dict[task]

            # Metric 1: Static Divergence
            stat_div = acc_t - f1_t

            record = {"Tasks": task, "Count": count, "Static_Div": stat_div}

            # Metric 2: Relative Delta Divergence (only valid if step > 0)
            if step > 0:
                delta_acc_rel = (acc_t - acc_0) / acc_0 if acc_0 != 0 else 0
                delta_f1_rel = (f1_t - f1_0) / f1_0 if f1_0 != 0 else 0
                record["Rel_Delta_Div"] = delta_acc_rel - delta_f1_rel

            task_records.append(record)

        df_step = pd.DataFrame(task_records)

        # 4. Calculate and Store Correlations
        if not df_step.empty and len(df_step) > 1:
            # Ensure variance before correlating to avoid NaN errors
            if df_step["Static_Div"].nunique() > 1 and df_step["Count"].nunique() > 1:
                steps_static.append(step)
                static_pearson.append(df_step["Static_Div"].corr(df_step["Count"], method="pearson"))
                static_spearman.append(df_step["Static_Div"].corr(df_step["Count"], method="spearman"))

            if step > 0 and "Rel_Delta_Div" in df_step.columns:
                if df_step["Rel_Delta_Div"].nunique() > 1 and df_step["Count"].nunique() > 1:
                    steps_rel.append(step)
                    rel_pearson.append(df_step["Rel_Delta_Div"].corr(df_step["Count"], method="pearson"))
                    rel_spearman.append(df_step["Rel_Delta_Div"].corr(df_step["Count"], method="spearman"))

    # 5. Plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Top Plot: Static Divergence
    axes[0].plot(steps_static, static_pearson, marker="o", label="Pearson", color="dodgerblue")
    axes[0].plot(steps_static, static_spearman, marker="s", linestyle="--", label="Spearman", color="crimson")
    axes[0].axhline(0, color="gray", linestyle="-", alpha=0.5)
    axes[0].set_title(r"Static Divergence ($Acc - F1$) vs. Dataset Count", fontsize=14)
    axes[0].set_ylabel("Correlation Coefficient", fontsize=12)
    axes[0].set_ylim(-1.1, 1.1)
    axes[0].legend()
    axes[0].grid(True, linestyle=":", alpha=0.7)

    # Bottom Plot: Relative Delta Divergence
    axes[1].plot(steps_rel, rel_pearson, marker="o", label="Pearson", color="dodgerblue")
    axes[1].plot(steps_rel, rel_spearman, marker="s", linestyle="--", label="Spearman", color="crimson")
    axes[1].axhline(0, color="gray", linestyle="-", alpha=0.5)
    axes[1].set_title(
        r"Relative Delta Divergence ($\Delta Acc_{rel} - \Delta F1_{rel}$) vs. Dataset Count", fontsize=14
    )
    axes[1].set_xlabel("RL Global Step", fontsize=12)
    axes[1].set_ylabel("Correlation Coefficient", fontsize=12)
    axes[1].set_ylim(-1.1, 1.1)
    axes[1].legend()
    axes[1].grid(True, linestyle=":", alpha=0.7)

    # Formatting and Save
    plt.tight_layout()
    output_plot = os.path.join(EVAL_DIR, "count_divergence_trend.png")
    plt.savefig(output_plot, dpi=300, bbox_inches="tight")
    print(f"Plot successfully saved to:\n{output_plot}")
    plt.show()


if __name__ == "__main__":
    main()
