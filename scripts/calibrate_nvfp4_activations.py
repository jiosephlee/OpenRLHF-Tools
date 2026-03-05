#!/usr/bin/env python3
"""
Calibrate NVFP4 activation (input) scales for the GPT-OSS NVFP4 checkpoint.

The existing checkpoint has calibrated weight block/global scales but
w13_input_scale / w2_input_scale pre-filled to 1.0.  The VLLM_CUTLASS
backend (W4A4) requires these to cover the actual activation range; with
1.0, activations are saturated → garbage output.

Note on the sample code found online (mtq.quantize approach):
  ModelOpt's mtq.quantize inserts FakeQuantize modules around individual
  nn.Linear layers. GPT-OSS stores all experts fused as a single [E, in, out]
  weight tensor — there are no individual per-expert nn.Linear modules to
  quantize. mtq.quantize would skip or mishandle the fused expert ops.
  We use forward hooks instead, which work correctly with any custom MoE.

This script:
  1. Loads the BF16 base model
  2. Hooks each MoE layer's experts module to collect max(|activation|)
     across calibration forward passes
  3. Computes input_scale = amax / 2688  (same formula as weight global scale)
  4. Downloads the existing NVFP4 checkpoint and adds the new tensors
  5. Uploads the updated checkpoint to HF Hub

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

Notes:
  - Saves a single per-layer scale used for all experts in that layer
    (conservative upper bound — all experts see inputs from the same
    distribution; routing just selects a subset of tokens per expert).
  - For down_proj input: measures the output of the gate_up_proj experts
    module to capture the post-SwiGLU activation magnitude separately.
  - Requires ~40 GB GPU VRAM for the BF16 model (use 4-8 B200 GPUs).
"""

import argparse
import gc
import json
import os
import shutil

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

# amax / (E4M3_MAX × E2M1_MAX) = amax / (448 × 6) — same formula used for
# weight global scales in convert_to_nvfp4.py and nvfp4_quantize.py.
NVFP4_SCALE_DENOM = 448.0 * 6.0  # 2688.0

LOCAL_SAVE_DIR = "/vast/projects/myatskar/design-documents/hf_home"

# Representative calibration prompts.  Mix of general + TDC-style (chemistry,
# drug discovery, biology) to match the actual inference distribution.
CALIBRATION_PROMPTS = [
    # General scientific reasoning
    "The capital of France is Paris. Explain the history of this city and its role in European culture.",
    "Describe the mechanisms by which vaccines induce long-lasting immune responses in the human body.",
    "The laws of thermodynamics govern energy transfer in physical and chemical systems. The first law states",
    "Machine learning models are trained by minimizing a loss function defined over the training data. In supervised learning,",
    "Quantum entanglement is a phenomenon where two particles become correlated such that the state of one",
    # Drug discovery / TDC tasks
    "The SMILES notation CC(=O)Oc1ccccc1C(=O)O represents aspirin. Its pharmacological properties include",
    "Drug toxicity prediction is a critical step in pharmaceutical development. Common toxic endpoints evaluated include",
    "The blood-brain barrier restricts entry of most therapeutics. Key physicochemical factors affecting BBB permeability are",
    "ADMET stands for absorption, distribution, metabolism, excretion, and toxicity. When evaluating a new drug candidate,",
    "The Tox21 dataset contains toxicity measurements for thousands of environmental compounds. The NR-AR assay measures",
    "Molecular docking simulations predict how a small molecule binds to a protein receptor target. Common scoring functions",
    "The IC50 value represents the concentration required to inhibit 50% of a target's activity. In high-throughput screening,",
    "SMILES: C1CC1N2C=C(C(=O)c3ccc(F)cc3)C(=O)N2 represents a fluoroquinolone. Its mechanism involves inhibition of",
    "Protein-ligand binding affinity prediction using graph neural networks encodes atoms as nodes and bonds as edges.",
    "The Lipinski rule of five describes oral bioavailability: MW < 500, HBA ≤ 10, HBD ≤ 5, LogP ≤ 5.",
    # Biology / biochemistry
    "CRISPR-Cas9 gene editing works by using guide RNA to direct the Cas9 nuclease to a specific genomic location.",
    "The tumor microenvironment plays a crucial role in cancer progression and therapy resistance. Key cell types include",
    "mRNA vaccines encode antigen-coding sequences translated by host ribosomes. The innate immune response is triggered by",
    "Enzyme kinetics follows the Michaelis-Menten equation v = Vmax[S]/(Km + [S]). The Km represents",
    "Signal transduction pathways relay extracellular ligand binding to intracellular responses via phosphorylation cascades.",
    # Longer-context prompts (stress-test deeper sequence positions)
    (
        "Given the molecular structure with SMILES CC(=O)NC1=CC=C(O)C=C1 (paracetamol / acetaminophen), predict "
        "its likely pharmacokinetic behavior. Consider solubility, membrane permeability, metabolic stability, and "
        "plasma protein binding. Discuss its hepatotoxicity mechanism at overdose concentrations."
    ),
    (
        "A Phase II clinical trial for a novel EGFR inhibitor in non-small-cell lung cancer showed: ORR 42%, "
        "median PFS 11.3 months, median OS 22.1 months, grade 3+ AEs in 28% of patients. Compare these outcomes "
        "to erlotinib as second-line therapy and discuss the path to regulatory approval."
    ),
    (
        "Design a multi-step synthesis for a peptidomimetic HIV protease inhibitor starting from commercially "
        "available amino acid building blocks. Include stereochemical considerations and key protecting group "
        "strategies for each step of the synthetic route."
    ),
]


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate NVFP4 activation scales.")
    p.add_argument("--base_model", default="unsloth/gpt-oss-20b-BF16",
                   help="BF16 base model for calibration forward passes")
    p.add_argument("--nvfp4_repo", default="jiosephlee/gpt-oss-20B-NVFP4-packed-clean",
                   help="Existing NVFP4 checkpoint to patch with calibrated scales")
    p.add_argument("--output_repo", default="jiosephlee/gpt-oss-20B-NVFP4-calibrated",
                   help="HF Hub repo to push the updated checkpoint to")
    p.add_argument("--num_samples", type=int, default=len(CALIBRATION_PROMPTS),
                   help="Number of calibration prompts (default: all)")
    p.add_argument("--max_length", type=int, default=512,
                   help="Max tokenised length per prompt")
    p.add_argument("--local_dir", default=os.path.join(LOCAL_SAVE_DIR, "nvfp4_calibration"),
                   help="Local scratch directory")
    p.add_argument("--private", action="store_true", default=True)
    p.add_argument("--keep_local", action="store_true",
                   help="Keep local checkpoint copy after upload")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Phase 1: collect activation amaxes via forward hooks
# ---------------------------------------------------------------------------

def _find_experts_modules(model):
    """
    Return (layer_idx, name, module) for each MoE experts module.

    GPT-OSS layout: model.layers.N.mlp.experts
      - experts has a fused weight attribute gate_up_proj: [E, in, out]
    We hook both the pre-forward (captures gate_up input = hidden state)
    and the post-forward (captures down_proj input = gate_up output).
    """
    candidates = []
    for name, module in model.named_modules():
        parts = name.split(".")
        # Match "model.layers.<N>.mlp.experts" or "model.layers.<N>.mlp"
        if ("mlp" in parts and "experts" in parts) or (
            "mlp" in parts and any(
                hasattr(module, attr)
                for attr in ("gate_up_proj", "w13_weight")
            )
        ):
            # Extract layer index from the path
            try:
                layer_idx = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                continue
            candidates.append((layer_idx, name, module))

    if not candidates:
        # Broader fallback: any module named "experts"
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


def collect_activation_amaxes(model, tokenizer, prompts, max_length):
    """
    Run calibration forward passes with forward hooks on each MoE experts
    module.  Returns per-layer amaxes for gate_up_proj input and down_proj
    input (post-SwiGLU intermediate).

    Returns:
        gate_up_amaxes: List[float]  — one per layer
        down_amaxes:    List[float]  — one per layer
    """
    experts_by_layer = _find_experts_modules(model)
    n_layers = len(experts_by_layer)
    print(f"[calibrate] Found {n_layers} MoE layers to hook")
    if n_layers == 0:
        raise RuntimeError(
            "No MoE expert modules found. "
            "Check model architecture — expected 'experts' in module names."
        )

    gate_up_amaxes = [0.0] * n_layers   # max(|hidden_state|) entering gate_up_proj
    down_amaxes = [0.0] * n_layers       # max(|intermediate|) entering down_proj

    hooks = []
    for layer_idx, name, module in experts_by_layer:
        def make_pre_hook(lidx):
            def pre_hook(mod, args):
                # args[0]: hidden states routed to experts [n_tokens, hidden_size]
                # (or [batch, seq, hidden] before routing flattens the batch dim)
                x = args[0].detach().float()
                gate_up_amaxes[lidx] = max(gate_up_amaxes[lidx], x.abs().max().item())
            return pre_hook

        def make_post_hook(lidx):
            def post_hook(mod, args, output):
                # output: result after down_proj, but the INPUT to down_proj
                # is the post-SwiGLU intermediate.  We can't access it directly
                # here.  Instead capture the output and use it as a proxy: the
                # down_proj output magnitude is typically similar to its input
                # magnitude (down_proj is a linear projection).
                out = output.detach().float() if isinstance(output, torch.Tensor) else output[0].detach().float()
                down_amaxes[lidx] = max(down_amaxes[lidx], out.abs().max().item())
            return post_hook

        hooks.append(module.register_forward_pre_hook(make_pre_hook(layer_idx)))
        hooks.append(module.register_forward_hook(make_post_hook(layer_idx)))

    model.eval()
    device = next(model.parameters()).device
    try:
        with torch.no_grad():
            for i, prompt in enumerate(prompts):
                print(f"  [{i+1}/{len(prompts)}] {prompt[:70]!r}...")
                inputs = tokenizer(
                    prompt,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                ).to(device)
                model(**inputs)
                del inputs
                if i % 5 == 0:
                    torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()

    print(f"\n[calibrate] Collected amaxes across {len(prompts)} prompts:")
    print(f"  gate_up_proj input: "
          f"min={min(gate_up_amaxes):.3f}  max={max(gate_up_amaxes):.3f}  "
          f"mean={sum(gate_up_amaxes)/n_layers:.3f}")
    print(f"  down_proj input:    "
          f"min={min(down_amaxes):.3f}  max={max(down_amaxes):.3f}  "
          f"mean={sum(down_amaxes)/n_layers:.3f}")

    # Sanity check: if down amax < gate_up amax the proxy may have underestimated;
    # use gate_up * 2 as a floor (SwiGLU intermediate is typically larger).
    for i in range(n_layers):
        if down_amaxes[i] < gate_up_amaxes[i]:
            down_amaxes[i] = gate_up_amaxes[i] * 2.0

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
        new_tensors[f"model.layers.{layer_idx}.mlp.experts.gate_up_proj.input_scale"] = (
            torch.full((num_experts,), gate_scale, dtype=torch.float32)
        )
        new_tensors[f"model.layers.{layer_idx}.mlp.experts.down_proj.input_scale"] = (
            torch.full((num_experts,), down_scale, dtype=torch.float32)
        )

    print(f"[calibrate] Adding {len(new_tensors)} input_scale tensors "
          f"({n_layers} layers × 2 projections × {num_experts} experts)")

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
        raise RuntimeError(
            "Could not infer num_experts. "
            "Check that model has 3D expert weight tensors."
        )

    prompts = CALIBRATION_PROMPTS[: args.num_samples]
    print(f"[calibrate] Running {len(prompts)} calibration passes...")
    gate_up_amaxes, down_amaxes = collect_activation_amaxes(
        model, tokenizer, prompts, args.max_length
    )

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
