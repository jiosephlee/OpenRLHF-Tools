#!/usr/bin/env python3
"""Convert a BF16 GPT-OSS model to NVFP4 format for vLLM deployment.

Mirrors ModelOpt's convert_oai_mxfp4_weight_only.py but produces NVFP4
weights (block_size=16, E4M3 scales, per-tensor FP32 global scale) instead
of MXFP4.

Saves locally, uploads to HuggingFace Hub, then deletes the local copy.

Two conversion backends:
  --backend modelopt  (recommended) Uses NVIDIA ModelOpt's NVFP4QTensor for
                      calibration-aware quantization. Requires `nvidia-modelopt`.
  --backend builtin   Uses our standalone quantize_to_nvfp4() implementation.
                      No extra dependencies beyond torch.

Usage:
  # ModelOpt backend (recommended):
  python scripts/convert_to_nvfp4.py \
      --model_path 2imi9/gpt-oss-20B-NVFP4A16-BF16 \
      --hub_repo_id jiosephlee/gpt-oss-20B-NVFP4-packed \
      --backend modelopt

  # Builtin backend (no modelopt dependency):
  python scripts/convert_to_nvfp4.py \
      --model_path 2imi9/gpt-oss-20B-NVFP4A16-BF16 \
      --hub_repo_id jiosephlee/gpt-oss-20B-NVFP4-packed \
      --backend builtin

  # Keep local copy (don't delete after upload):
  python scripts/convert_to_nvfp4.py \
      --model_path 2imi9/gpt-oss-20B-NVFP4A16-BF16 \
      --hub_repo_id jiosephlee/gpt-oss-20B-NVFP4-packed \
      --keep_local

  # With LoRA adapter merging:
  python scripts/convert_to_nvfp4.py \
      --base_path 2imi9/gpt-oss-20B-NVFP4A16-BF16 \
      --lora_path ./my-lora-adapter \
      --hub_repo_id jiosephlee/gpt-oss-20B-NVFP4-merged

After conversion, load in vLLM with:
  vllm serve jiosephlee/gpt-oss-20B-NVFP4-packed
  # vLLM auto-detects quantization from hf_quant_config.json
"""

import argparse
import gc
import json
import os
import shutil

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LOCAL_SAVE_DIR = "/vast/projects/myatskar/design-documents/hf_home"


def _convert_nvfp4_modelopt(model, block_size=16):
    """Convert MoE expert weights to NVFP4 using ModelOpt's NVFP4QTensor."""
    from modelopt.torch.quantization.qtensor import NVFP4QTensor

    new_state_dict = {}

    for name, param in model.state_dict().items():
        # Only convert expert weights, skip bias and other modules
        if "experts" in name and "bias" not in name:
            # Transpose: [E, in, out] -> [E, out, in] (checkpoint convention)
            param = param.transpose(-1, -2).contiguous()

            packed_list = []
            scale_list = []
            global_scale_list = []

            for expert in param:
                quantized, scales, global_scale = NVFP4QTensor.quantize(expert, block_size=block_size)
                packed_list.append(quantized._quantized_data)
                scale_list.append(scales)
                global_scale_list.append(global_scale)

            packed = torch.stack(packed_list)  # [E, out, K//2]
            scales = torch.stack(scale_list)  # [E, out, K//group_size]
            global_scales = torch.stack(global_scale_list)  # [E]

            # Save as 3D — vLLM expects [E, out, K//2] (not 4D with block sub-dim)
            new_state_dict[f"{name}_blocks"] = packed.cpu()
            new_state_dict[f"{name}_scale"] = scales.cpu()
            new_state_dict[f"{name}_scale_2"] = global_scales.cpu()

            del param, packed, scales, global_scales
            torch.cuda.empty_cache()
            gc.collect()
        else:
            new_state_dict[name] = param

    return new_state_dict


def _convert_nvfp4_builtin(model, block_size=16):
    """Convert MoE expert weights to NVFP4 using our standalone implementation."""
    from openrlhf.utils.nvfp4_quantize import quantize_to_nvfp4

    new_state_dict = {}

    for name, param in model.state_dict().items():
        if "experts" in name and "bias" not in name:
            # Transpose: [E, in, out] -> [E, out, in]
            param_t = param.transpose(-1, -2).contiguous()

            packed_list = []
            scale_list = []
            global_scale_list = []

            for i in range(param_t.shape[0]):
                expert = param_t[i].cuda() if not param_t.is_cuda else param_t[i]
                packed, scales, global_scale = quantize_to_nvfp4(expert, block_size=block_size)
                packed_list.append(packed.cpu())
                scale_list.append(scales.cpu())
                global_scale_list.append(global_scale.cpu())

            packed = torch.stack(packed_list)  # [E, out, K//2]
            scales = torch.stack(scale_list)  # [E, out, K//group_size]
            global_scales = torch.stack(global_scale_list)  # [E]

            # Save as 3D — vLLM expects [E, out, K//2] (not 4D with block sub-dim)
            new_state_dict[f"{name}_blocks"] = packed
            new_state_dict[f"{name}_scale"] = scales
            new_state_dict[f"{name}_scale_2"] = global_scales

            del param_t, packed, scales, global_scales
            torch.cuda.empty_cache()
            gc.collect()
        else:
            new_state_dict[name] = param

    return new_state_dict


def convert_and_save(model, tokenizer, output_path, backend="modelopt", block_size=16):
    """Quantize expert weights to NVFP4 and save the checkpoint."""
    print(f"Converting to NVFP4 using backend={backend}, block_size={block_size}")

    if backend == "modelopt":
        quantized_state_dict = _convert_nvfp4_modelopt(model, block_size)
    elif backend == "builtin":
        quantized_state_dict = _convert_nvfp4_builtin(model, block_size)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Save converted weights
    model.save_pretrained(output_path, state_dict=quantized_state_dict)

    # Update config.json with quantization_config
    config_path = os.path.join(output_path, "config.json")
    with open(config_path) as f:
        config_data = json.load(f)

    config_data["quantization_config"] = {
        "modules_to_not_convert": [
            "model.layers.*.self_attn",
            "model.layers.*.mlp.router",
            "model.embed_tokens",
            "lm_head",
        ],
        "quant_method": "modelopt",
        "quant_algo": "NVFP4",
    }
    config_data.pop("torch_dtype", None)

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)

    # Create hf_quant_config.json (required by vLLM's modelopt_fp4)
    hf_quant_config = {
        "producer": {"name": "modelopt", "version": "0.35.0"},
        "quantization": {
            "quant_algo": "NVFP4",
            "kv_cache_quant_algo": None,
            "group_size": block_size,
            "exclude_modules": ["lm_head"],
        },
    }
    hf_quant_config_path = os.path.join(output_path, "hf_quant_config.json")
    with open(hf_quant_config_path, "w") as f:
        json.dump(hf_quant_config, f, indent=4)

    # Save tokenizer
    tokenizer.save_pretrained(output_path)

    print(f"NVFP4 checkpoint saved to: {output_path}")
    print(f"  - config.json: updated with quantization_config")
    print(f"  - hf_quant_config.json: created for vLLM modelopt_fp4")


def upload_to_hub(local_path, repo_id, private=True):
    """Upload the converted checkpoint to HuggingFace Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    print(f"Uploading {local_path} to https://huggingface.co/{repo_id} ...")
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(
        folder_path=local_path,
        repo_id=repo_id,
        repo_type="model",
    )
    print(f"Upload complete: https://huggingface.co/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Convert a BF16 GPT-OSS model to NVFP4 format for vLLM.")
    parser.add_argument(
        "--model_path",
        type=str,
        help="Path to the BF16 model (HF hub ID or local path).",
    )
    parser.add_argument(
        "--lora_path",
        type=str,
        help="Path to LoRA adapter weights. Mutually exclusive with --model_path.",
    )
    parser.add_argument(
        "--base_path",
        type=str,
        help="Path to base model for LoRA merging. Required if --lora_path is set.",
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        required=True,
        help="HuggingFace Hub repo ID to upload to (e.g., 'jiosephlee/gpt-oss-20B-NVFP4').",
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=LOCAL_SAVE_DIR,
        help=f"Local directory for temporary checkpoint storage (default: {LOCAL_SAVE_DIR}).",
    )
    parser.add_argument(
        "--keep_local",
        action="store_true",
        help="Keep the local copy after uploading to HF Hub.",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the HF Hub repo public (default: private).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="modelopt",
        choices=["modelopt", "builtin"],
        help="Quantization backend: 'modelopt' (recommended) or 'builtin'.",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=16,
        help="NVFP4 block size (default: 16).",
    )
    args = parser.parse_args()

    # Derive local output path from hub repo ID
    repo_name = args.hub_repo_id.replace("/", "--")
    output_path = os.path.join(args.local_dir, repo_name)
    os.makedirs(output_path, exist_ok=True)

    # Load model
    kwargs = {"device_map": "auto", "torch_dtype": torch.bfloat16, "trust_remote_code": True}

    if args.lora_path:
        assert args.model_path is None, "Cannot specify both --model_path and --lora_path."
        assert args.base_path is not None, "--base_path is required when using --lora_path."
        model_path = args.base_path
    else:
        assert args.model_path is not None, "Must specify --model_path or --lora_path."
        model_path = args.model_path

    print(f"Loading model from: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)

    if args.lora_path:
        from peft import PeftModel

        print(f"Merging LoRA adapter from: {args.lora_path}")
        model = PeftModel.from_pretrained(model, args.lora_path)
        model = model.merge_and_unload()
        torch.cuda.empty_cache()
        gc.collect()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Step 1: Convert and save locally
    convert_and_save(model, tokenizer, output_path, args.backend, args.block_size)

    # Free model memory before upload
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # Step 2: Upload to HuggingFace Hub
    upload_to_hub(output_path, args.hub_repo_id, private=not args.public)

    # Step 3: Delete local copy
    if not args.keep_local:
        print(f"Deleting local copy: {output_path}")
        shutil.rmtree(output_path)
        print("Local copy deleted.")
    else:
        print(f"Local copy kept at: {output_path}")


if __name__ == "__main__":
    main()
