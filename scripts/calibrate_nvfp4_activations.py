#!/usr/bin/env python3
"""
Calibrate NVFP4 activation (input) scales for the GPT-OSS NVFP4 checkpoint.

The existing checkpoint has calibrated weight block/global scales but
w13_input_scale / w2_input_scale pre-filled to 1.0.  The VLLM_CUTLASS
backend (W4A4) requires these to cover the actual activation range; with
1.0, activations are saturated → garbage output.

This script:
  1. Loads the BF16 base model
  2. Monkey-patches each GptOssExperts.forward to capture:
     - gate_up_proj input amax (hidden_states entering the expert)
     - down_proj input amax (gated_output, the post-SwiGLU intermediate)
  3. Runs calibration on actual TDC dataset prompts (matching inference
     distribution) plus a few generic prompts for coverage
  4. Computes input_scale = amax / 2688  (same formula as weight global scale)
  5. Downloads the existing NVFP4 checkpoint and adds the new tensors
  6. Uploads the updated checkpoint to HF Hub

Previous version used post_hook on the experts module output as a proxy for
down_proj input — this is WRONG because the module output is the routing-
weighted sum of expert outputs, not the post-SwiGLU intermediate that is
the actual input to down_proj.  The fix: monkey-patch forward() to capture
gated_output directly inside the per-expert loop.

New checkpoint keys (picked up by hf_to_vllm_mapper in gpt_oss.py):
  model.layers.N.mlp.experts.gate_up_proj.input_scale  [E]  → w13_input_scale
  model.layers.N.mlp.experts.down_proj.input_scale     [E]  → w2_input_scale

Usage:
    module load cuda/12.8.1
    conda activate /vast/projects/myatskar/design-documents/conda_env/openrlhf
    python scripts/calibrate_nvfp4_activations.py \\
        --base_model unsloth/gpt-oss-20b-BF16 \\
        --nvfp4_repo jiosephlee/gpt-oss-20B-NVFP4-packed-clean \\
        --output_repo jiosephlee/gpt-oss-20B-NVFP4-calibrated \\
        --num_samples 512

    # Use only TDC data (skip generic prompts):
    python scripts/calibrate_nvfp4_activations.py \\
        --tdc_data_dir data/tdc/openai_format_gpt_oss \\
        --num_samples 256

Notes:
  - Saves a single per-layer scale used for all experts in that layer
    (conservative upper bound — all experts see inputs from the same
    distribution; routing just selects a subset of tokens per expert).
  - Requires ~40 GB GPU VRAM for the BF16 model (use 4-8 B200 GPUs).
"""

import argparse
import gc
import glob
import json
import os
import random
import shutil
from functools import wraps

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# amax / (E4M3_MAX × E2M1_MAX) = amax / (448 × 6) — same formula used for
# weight global scales in convert_to_nvfp4.py and nvfp4_quantize.py.
NVFP4_SCALE_DENOM = 448.0 * 6.0  # 2688.0

LOCAL_SAVE_DIR = "/vast/projects/myatskar/design-documents/hf_home"

# A small set of generic prompts for coverage beyond TDC distribution.
# The bulk of calibration should come from actual TDC data via --tdc_data_dir.
GENERIC_PROMPTS = [
    "The capital of France is Paris. Explain the history of this city and its role in European culture.",
    "Describe the mechanisms by which vaccines induce long-lasting immune responses in the human body.",
    "Machine learning models are trained by minimizing a loss function defined over the training data. In supervised learning,",
    "Quantum entanglement is a phenomenon where two particles become correlated such that the state of one",
    "The SMILES notation CC(=O)Oc1ccccc1C(=O)O represents aspirin. Its pharmacological properties include",
    "Drug toxicity prediction is a critical step in pharmaceutical development. Common toxic endpoints evaluated include",
    "The blood-brain barrier restricts entry of most therapeutics. Key physicochemical factors affecting BBB permeability are",
    "ADMET stands for absorption, distribution, metabolism, excretion, and toxicity. When evaluating a new drug candidate,",
    "The Tox21 dataset contains toxicity measurements for thousands of environmental compounds. The NR-AR assay measures",
    "CRISPR-Cas9 gene editing works by using guide RNA to direct the Cas9 nuclease to a specific genomic location.",
]


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate NVFP4 activation scales.")
    p.add_argument(
        "--base_model", default="unsloth/gpt-oss-20b-BF16", help="BF16 base model for calibration forward passes"
    )
    p.add_argument(
        "--nvfp4_repo",
        default="jiosephlee/gpt-oss-20B-NVFP4-packed-clean",
        help="Existing NVFP4 checkpoint to patch with calibrated scales",
    )
    p.add_argument(
        "--output_repo",
        default="jiosephlee/gpt-oss-20B-NVFP4-calibrated",
        help="HF Hub repo to push the updated checkpoint to",
    )
    p.add_argument(
        "--tdc_data_dir",
        default=None,
        help="Directory with TDC JSONL files (e.g., data/tdc/openai_format_gpt_oss). "
        "If provided, samples actual TDC prompts for calibration.",
    )
    p.add_argument(
        "--num_samples", type=int, default=0, help="Max calibration prompts (0 = use all available, default: all)"
    )
    p.add_argument("--max_length", type=int, default=512, help="Max tokenised length per prompt")
    p.add_argument(
        "--local_dir", default=os.path.join(LOCAL_SAVE_DIR, "nvfp4_calibration"), help="Local scratch directory"
    )
    p.add_argument("--private", action="store_true", default=True)
    p.add_argument("--keep_local", action="store_true", help="Keep local checkpoint copy after upload")
    p.add_argument("--seed", type=int, default=42, help="Random seed for TDC prompt sampling")
    p.add_argument("--skip_generic", action="store_true", help="Skip generic prompts, use only TDC data")
    return p.parse_args()


# ---------------------------------------------------------------------------
# TDC prompt loading
# ---------------------------------------------------------------------------


def load_tdc_prompts(tdc_data_dir, num_samples, seed=42, max_per_task=5000):
    """
    Load calibration prompts from TDC JSONL files.

    Samples uniformly across all available task train files to get a
    representative distribution of SMILES strings and prompt templates.
    Tasks with more than max_per_task samples are subsampled to avoid
    large datasets (e.g. Tox21 54k, HIV 29k) dominating calibration.

    Returns:
        List[str] — prompt strings ready for tokenization
    """
    rng = random.Random(seed)

    # Resolve relative paths from the repo root (script lives in scripts/)
    if not os.path.isabs(tdc_data_dir):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        tdc_data_dir = os.path.join(repo_root, tdc_data_dir)

    # Find all train JSONL files
    pattern = os.path.join(tdc_data_dir, "*_train.jsonl")
    train_files = sorted(glob.glob(pattern))
    if not train_files:
        raise FileNotFoundError(f"No *_train.jsonl files found in {tdc_data_dir}. Tried pattern: {pattern}")

    # Load prompts per task, cap large datasets
    all_prompts = []
    task_counts = {}
    for fpath in train_files:
        task_name = os.path.basename(fpath).replace("_train.jsonl", "")
        task_prompts = []
        with open(fpath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                messages = record.get("messages", [])
                for msg in messages:
                    if msg["role"] == "user":
                        task_prompts.append(msg["content"])
                        break

        raw_count = len(task_prompts)
        if raw_count > max_per_task:
            rng.shuffle(task_prompts)
            task_prompts = task_prompts[:max_per_task]
            print(f"  {task_name}: {raw_count} → capped to {max_per_task}")
        task_counts[task_name] = len(task_prompts)
        all_prompts.extend(task_prompts)

    print(f"[calibrate] Loaded {len(all_prompts)} TDC prompts from {len(train_files)} tasks:")
    for task, count in sorted(task_counts.items()):
        print(f"  {task}: {count}")

    # Sample if a global cap was requested, otherwise use all
    if num_samples > 0 and num_samples < len(all_prompts):
        rng.shuffle(all_prompts)
        all_prompts = all_prompts[:num_samples]
    else:
        rng.shuffle(all_prompts)  # Shuffle for variety across tasks

    return all_prompts


# ---------------------------------------------------------------------------
# Phase 1: collect activation amaxes via monkey-patched forward
# ---------------------------------------------------------------------------


def _find_experts_modules(model):
    """
    Return (layer_idx, name, module) for each MoE experts module.

    GPT-OSS layout: model.layers.N.mlp.experts
      - experts has a fused weight attribute gate_up_proj: [E, in, out]
    """
    candidates = []
    for name, module in model.named_modules():
        parts = name.split(".")
        if ("mlp" in parts and "experts" in parts) or (
            "mlp" in parts and any(hasattr(module, attr) for attr in ("gate_up_proj", "w13_weight"))
        ):
            try:
                layer_idx = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                continue
            candidates.append((layer_idx, name, module))

    if not candidates:
        for name, module in model.named_modules():
            if name.split(".")[-1] == "experts":
                parts = name.split(".")
                try:
                    layer_idx = int(parts[parts.index("layers") + 1])
                except (ValueError, IndexError):
                    continue
                candidates.append((layer_idx, name, module))

    # Deduplicate by layer_idx, prefer the most specific path (deepest)
    seen = {}
    for layer_idx, name, module in sorted(candidates, key=lambda x: len(x[1])):
        seen[layer_idx] = (name, module)
    return [(idx, name, mod) for idx, (name, mod) in sorted(seen.items())]


def _monkey_patch_experts_forward(module, layer_idx, gate_up_amaxes, down_amaxes):
    """
    Monkey-patch GptOssExperts.forward to capture activation amaxes for both
    gate_up_proj input (hidden_states) and down_proj input (gated_output).

    The key insight: down_proj input = gated_output (the post-SwiGLU
    intermediate), which is computed INSIDE the per-expert loop. A post_hook
    on the module only sees the final weighted output — totally different
    distribution. We must intercept inside the loop.
    """
    original_forward = module.forward

    @wraps(original_forward)
    def patched_forward(hidden_states, router_indices=None, routing_weights=None):
        import torch.nn.functional as F

        next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)
        with torch.no_grad():
            expert_mask = F.one_hot(router_indices, num_classes=module.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == module.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]

            # --- Capture gate_up_proj input amax ---
            with torch.no_grad():
                # Reduce in native dtype first, then convert scalar to float
                # to avoid bf16→float32 bulk cast issues on multi-GPU setups
                amax_val = current_state.detach().abs().max().float().item()
                gate_up_amaxes[layer_idx] = max(gate_up_amaxes[layer_idx], amax_val)

            gate_up = current_state @ module.gate_up_proj[expert_idx] + module.gate_up_proj_bias[expert_idx]
            gated_output = module._apply_gate(gate_up)

            # --- Capture down_proj input amax (the ACTUAL intermediate) ---
            with torch.no_grad():
                amax_val = gated_output.detach().abs().max().float().item()
                down_amaxes[layer_idx] = max(down_amaxes[layer_idx], amax_val)

            out = gated_output @ module.down_proj[expert_idx] + module.down_proj_bias[expert_idx]
            weighted_output = out * routing_weights[token_idx, top_k_pos, None]
            next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))

        return next_states

    module.forward = patched_forward
    return original_forward  # Return original so we can restore later


def collect_activation_amaxes(model, tokenizer, prompts, max_length):
    """
    Run calibration forward passes with monkey-patched experts.forward on
    each MoE layer.  Captures the ACTUAL intermediate activations:
      - gate_up_proj input = hidden_states routed to each expert
      - down_proj input = gated_output (post-SwiGLU), computed inside the
        per-expert loop

    Returns:
        gate_up_amaxes: List[float]  — one per layer
        down_amaxes:    List[float]  — one per layer
    """
    experts_by_layer = _find_experts_modules(model)
    n_layers = len(experts_by_layer)
    print(f"[calibrate] Found {n_layers} MoE layers to patch")
    if n_layers == 0:
        raise RuntimeError(
            "No MoE expert modules found. Check model architecture — expected 'experts' in module names."
        )

    gate_up_amaxes = [0.0] * n_layers
    down_amaxes = [0.0] * n_layers

    # Monkey-patch all experts modules
    originals = []
    for layer_idx, name, module in experts_by_layer:
        orig = _monkey_patch_experts_forward(module, layer_idx, gate_up_amaxes, down_amaxes)
        originals.append((module, orig))

    model.eval()
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                print(f"  [{i + 1}/{len(prompts)}] {prompt[:80]!r}...")
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                ).to(device)
                model(**inputs)
                del inputs
                if i % 10 == 0:
                    torch.cuda.empty_cache()
    finally:
        # Restore original forward methods
        for module, orig in originals:
            module.forward = orig

    print(f"\n[calibrate] Collected amaxes across {len(prompts)} prompts:")
    for i in range(n_layers):
        print(f"  layer {i:2d}: gate_up_input={gate_up_amaxes[i]:.4f}  down_input={down_amaxes[i]:.4f}")
    print(
        f"\n  gate_up_proj input:  min={min(gate_up_amaxes):.3f}  "
        f"max={max(gate_up_amaxes):.3f}  mean={sum(gate_up_amaxes) / n_layers:.3f}"
    )
    print(
        f"  down_proj input:     min={min(down_amaxes):.3f}  "
        f"max={max(down_amaxes):.3f}  mean={sum(down_amaxes) / n_layers:.3f}"
    )

    # Sanity check: all amaxes should be > 0
    for i in range(n_layers):
        if gate_up_amaxes[i] == 0.0:
            print(f"  WARNING: gate_up_amaxes[{i}] is 0 — layer may not have been hit")
        if down_amaxes[i] == 0.0:
            print(f"  WARNING: down_amaxes[{i}] is 0 — layer may not have been hit")

    return gate_up_amaxes, down_amaxes


# ---------------------------------------------------------------------------
# Phase 2: patch the NVFP4 checkpoint with calibrated input_scale tensors
# ---------------------------------------------------------------------------


def patch_checkpoint(nvfp4_repo, checkpoint_dir, gate_up_amaxes, down_amaxes, num_experts):
    """
    Add gate_up_proj.input_scale / down_proj.input_scale tensors to the
    existing NVFP4 checkpoint shards.

    Shape: [E] (one scalar per expert, same value — per-layer calibration).
    The _load_weights_nvfp4 loader expands [E] → [E, 2] for w13_input_scale.
    """
    import safetensors.torch
    from safetensors import safe_open

    n_layers = len(gate_up_amaxes)
    new_tensors = {}
    for layer_idx in range(n_layers):
        gate_scale = float(gate_up_amaxes[layer_idx]) / NVFP4_SCALE_DENOM
        down_scale = float(down_amaxes[layer_idx]) / NVFP4_SCALE_DENOM
        # [E] — broadcast same scale across all experts in this layer.
        new_tensors[f"model.layers.{layer_idx}.mlp.experts.gate_up_proj.input_scale"] = torch.full(
            (num_experts,), gate_scale, dtype=torch.float32
        )
        new_tensors[f"model.layers.{layer_idx}.mlp.experts.down_proj.input_scale"] = torch.full(
            (num_experts,), down_scale, dtype=torch.float32
        )

    print(f"\n[calibrate] Calibrated scales (input_scale = amax / {NVFP4_SCALE_DENOM}):")
    for layer_idx in range(n_layers):
        gs = float(gate_up_amaxes[layer_idx]) / NVFP4_SCALE_DENOM
        ds = float(down_amaxes[layer_idx]) / NVFP4_SCALE_DENOM
        print(f"  layer {layer_idx:2d}: gate_up_input_scale={gs:.6e}  down_input_scale={ds:.6e}")

    print(
        f"\n[calibrate] Adding {len(new_tensors)} input_scale tensors "
        f"({n_layers} layers × 2 projections × {num_experts} experts)"
    )

    index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    single_path = os.path.join(checkpoint_dir, "model.safetensors")

    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        # Append to the last shard (alphabetically — typically smallest).
        shards = sorted(set(index["weight_map"].values()))
        target_shard = shards[-1]
        target_path = os.path.join(checkpoint_dir, target_shard)
        print(f"[calibrate] Appending to shard: {target_shard}")

        existing = {}
        with safe_open(target_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                existing[key] = f.get_tensor(key)

        safetensors.torch.save_file({**existing, **new_tensors}, target_path)

        for key in new_tensors:
            index["weight_map"][key] = target_shard
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    elif os.path.exists(single_path):
        existing = {}
        with safe_open(single_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                existing[key] = f.get_tensor(key)
        safetensors.torch.save_file({**existing, **new_tensors}, single_path)

    else:
        raise FileNotFoundError(f"No safetensors checkpoint found in {checkpoint_dir}")

    print(f"[calibrate] Checkpoint patched.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    os.makedirs(args.local_dir, exist_ok=True)

    # ---- Build calibration prompts ----
    prompts = []

    # Load TDC prompts if data dir is provided
    if args.tdc_data_dir:
        tdc_prompts = load_tdc_prompts(args.tdc_data_dir, args.num_samples, seed=args.seed)
        prompts.extend(tdc_prompts)
        if not args.skip_generic:
            prompts.extend(GENERIC_PROMPTS)
        print(
            f"[calibrate] Using {len(tdc_prompts)} TDC prompts + "
            f"{len(GENERIC_PROMPTS) if not args.skip_generic else 0} generic prompts "
            f"= {len(prompts)} total"
        )
    else:
        # Fallback: use generic prompts only (not recommended)
        prompts = list(GENERIC_PROMPTS)
        print(
            f"[calibrate] WARNING: No --tdc_data_dir provided. Using only {len(prompts)} "
            f"generic prompts. For best results, pass --tdc_data_dir data/tdc/openai_format_gpt_oss"
        )

    # ---- Phase 1: calibration forward passes ----
    print(f"\n[calibrate] Loading BF16 model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"[calibrate] Model loaded ({n_params:.1f}B params)")

    # Detect num_experts from parameter shapes.
    num_experts = None
    for name, param in model.named_parameters():
        if "experts" in name and ("gate_up_proj" in name or "w13" in name) and param.ndim == 3:
            num_experts = param.shape[0]
            print(f"[calibrate] Detected num_experts={num_experts} from {name} {tuple(param.shape)}")
            break
    if num_experts is None:
        raise RuntimeError("Could not infer num_experts. Check that model has 3D expert weight tensors.")

    print(f"[calibrate] Running {len(prompts)} calibration passes...")
    gate_up_amaxes, down_amaxes = collect_activation_amaxes(model, tokenizer, prompts, args.max_length)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ---- Phase 2: download + patch checkpoint ----
    checkpoint_dir = os.path.join(args.local_dir, "patched_checkpoint")
    if os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)

    print(f"\n[calibrate] Downloading {args.nvfp4_repo}...")
    snapshot_download(args.nvfp4_repo, local_dir=checkpoint_dir, ignore_patterns=["*.bin"])

    patch_checkpoint(
        nvfp4_repo=args.nvfp4_repo,
        checkpoint_dir=checkpoint_dir,
        gate_up_amaxes=gate_up_amaxes,
        down_amaxes=down_amaxes,
        num_experts=num_experts,
    )

    # ---- Phase 3: upload ----
    print(f"\n[calibrate] Uploading to {args.output_repo}...")
    api = HfApi()
    api.create_repo(args.output_repo, exist_ok=True, private=args.private)
    api.upload_folder(
        folder_path=checkpoint_dir,
        repo_id=args.output_repo,
        repo_type="model",
    )
    print(f"[calibrate] Done → https://huggingface.co/{args.output_repo}")

    if not args.keep_local:
        shutil.rmtree(checkpoint_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
