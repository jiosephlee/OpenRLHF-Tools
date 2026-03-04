#!/usr/bin/env python3
"""
Minimal NVFP4 vLLM inference sanity check.

Loads the GPT-OSS NVFP4 checkpoint and generates a short response to confirm
vLLM can produce tokens at all, before investing in full RLHF training.

Usage:
    # Basic (uses VLLM_CUTLASS, no FlashInfer):
    VLLM_USE_FLASHINFER_MOE_FP4=0 python scripts/test_nvfp4_inference.py

    # With FlashInfer (to compare):
    python scripts/test_nvfp4_inference.py

    # Override model:
    MODEL=jiosephlee/gpt-oss-20B-NVFP4-packed-clean python scripts/test_nvfp4_inference.py

Run on a GPU node with the CUDA module loaded, e.g.:
    module load cuda/12.8.1
    conda activate /vast/projects/myatskar/design-documents/conda_env/openrlhf
    VLLM_USE_FLASHINFER_MOE_FP4=0 python scripts/test_nvfp4_inference.py
"""

import os
import subprocess
import sys
import time

MODEL = os.environ.get("MODEL", "jiosephlee/gpt-oss-20B-NVFP4-packed-clean")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "50"))
PROMPT = os.environ.get("PROMPT", "The capital of France is")

# Preflight: check CUDA is accessible using nvidia-smi — avoids initializing the CUDA
# context in the main process (which would cause vLLM's forked EngineCore subprocess
# to fail with "Cannot re-initialize CUDA in forked subprocess").
_smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
if _smi.returncode != 0:
    print(
        f"ERROR: nvidia-smi failed (returncode={_smi.returncode}): {_smi.stderr.strip()}\n"
        "Make sure you are on a GPU node and have loaded the CUDA module:\n"
        "  module load cuda/12.8.1\n"
        "  conda activate /vast/projects/myatskar/design-documents/conda_env/openrlhf\n"
        "  VLLM_USE_FLASHINFER_MOE_FP4=0 python scripts/test_nvfp4_inference.py",
        file=sys.stderr,
    )
    sys.exit(1)
print(f"[test_nvfp4_inference] GPUs detected:\n  " + _smi.stdout.strip().replace("\n", "\n  "))

flashinfer_disabled = os.environ.get("VLLM_USE_FLASHINFER_MOE_FP4", "") == "0"
print(f"[test_nvfp4_inference] Model: {MODEL}")
print(f"[test_nvfp4_inference] FlashInfer MoE FP4: {'DISABLED (VLLM_CUTLASS)' if flashinfer_disabled else 'enabled (default)'}")
print(f"[test_nvfp4_inference] Prompt: {PROMPT!r}")
print(f"[test_nvfp4_inference] Max new tokens: {MAX_NEW_TOKENS}")
print()

try:
    from vllm import LLM, SamplingParams
except ImportError:
    print("ERROR: vllm not installed. Activate the correct conda env.", file=sys.stderr)
    sys.exit(1)

print("[test_nvfp4_inference] Initializing vLLM engine...")
t0 = time.time()
try:
    llm = LLM(
        model=MODEL,
        trust_remote_code=True,
        max_model_len=512,
        gpu_memory_utilization=0.8,
        tensor_parallel_size=1,
    )
except Exception as e:
    print(f"ERROR: vLLM initialization failed: {e}", file=sys.stderr)
    sys.exit(1)

load_time = time.time() - t0
print(f"[test_nvfp4_inference] Engine initialized in {load_time:.1f}s")

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=MAX_NEW_TOKENS,
)

print(f"[test_nvfp4_inference] Generating {MAX_NEW_TOKENS} tokens...")
t1 = time.time()
try:
    outputs = llm.generate([PROMPT], sampling_params)
except Exception as e:
    print(f"ERROR: Generation failed: {e}", file=sys.stderr)
    sys.exit(1)

gen_time = time.time() - t1
output_text = outputs[0].outputs[0].text
n_tokens = len(outputs[0].outputs[0].token_ids)

print()
print("=" * 60)
print(f"PROMPT:  {PROMPT!r}")
print(f"OUTPUT:  {output_text!r}")
print(f"Tokens:  {n_tokens}")
print(f"Time:    {gen_time:.2f}s  ({n_tokens/gen_time:.1f} tok/s)")
print("=" * 60)

if not output_text.strip():
    print("\nWARNING: Output is empty — possible hang/garbage or all-whitespace generation.", file=sys.stderr)
    sys.exit(1)

print("\n[test_nvfp4_inference] SUCCESS: model generated non-empty output.")
