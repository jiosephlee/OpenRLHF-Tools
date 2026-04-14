#!/usr/bin/env python3
"""Minimal vLLM-only repro for gpt-oss generation issues."""

import os
import platform
import subprocess
import sys
import time


MODEL = os.environ.get("MODEL", "unsloth/gpt-oss-20b-BF16")
MOE_BACKEND = os.environ.get("MOE_BACKEND", "auto")
DTYPE = os.environ.get("DTYPE", "auto")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.8"))
TOP_P = float(os.environ.get("TOP_P", "0.95"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "64"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "1024"))
TENSOR_PARALLEL_SIZE = int(os.environ.get("TENSOR_PARALLEL_SIZE", "1"))
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.85"))
SEED = int(os.environ.get("SEED", "0"))
TRUST_REMOTE_CODE = os.environ.get("TRUST_REMOTE_CODE", "1") != "0"
ENFORCE_EAGER = os.environ.get("ENFORCE_EAGER", "0") == "1"

PROMPTS = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]


def print_env() -> None:
    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")

    try:
        import torch

        print(f"torch: {torch.__version__}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
        if torch.cuda.is_available():
            for idx in range(torch.cuda.device_count()):
                print(f"cuda:{idx}: {torch.cuda.get_device_name(idx)}")
    except Exception as exc:
        print(f"torch import failed: {exc}")

    try:
        import vllm

        print(f"vllm: {vllm.__version__}")
    except Exception as exc:
        print(f"vllm import failed: {exc}")

    smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    if smi.returncode == 0:
        print("nvidia-smi -L:")
        print(smi.stdout.strip())
    else:
        stderr = smi.stderr.strip() or smi.stdout.strip() or "not available"
        print(f"nvidia-smi -L failed: {stderr}")


def main() -> int:
    print_env()
    print()
    print(f"Model: {MODEL}")
    print(f"Prompts: {len(PROMPTS)}")
    print(
        f"Sampling: temperature={TEMPERATURE}, top_p={TOP_P}, max_tokens={MAX_TOKENS}"
    )
    print(
        "Engine:"
        f" tp={TENSOR_PARALLEL_SIZE}, dtype={DTYPE}, moe_backend={MOE_BACKEND},"
        f" max_model_len={MAX_MODEL_LEN},"
        f" gpu_memory_utilization={GPU_MEMORY_UTILIZATION},"
        f" enforce_eager={ENFORCE_EAGER}"
    )
    print()

    try:
        import torch
    except Exception as exc:
        print(f"ERROR: torch import failed: {exc}", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("ERROR: No CUDA devices are visible in this shell.", file=sys.stderr)
        return 2

    try:
        from vllm import LLM, SamplingParams
    except Exception as exc:
        print(f"ERROR: vLLM import failed: {exc}", file=sys.stderr)
        return 1

    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )

    print("Initializing LLM...")
    start = time.time()
    llm = LLM(
        model=MODEL,
        trust_remote_code=TRUST_REMOTE_CODE,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        dtype=DTYPE,
        moe_backend=MOE_BACKEND,
        seed=SEED,
        enforce_eager=ENFORCE_EAGER,
    )
    print(f"LLM initialized in {time.time() - start:.1f}s")

    print("Generating...")
    start = time.time()
    outputs = llm.generate(PROMPTS, sampling_params)
    print(f"Generation finished in {time.time() - start:.1f}s")
    print()

    for output in outputs:
        generated = output.outputs[0]
        print(f"Prompt: {output.prompt!r}")
        print(f"Text:   {generated.text!r}")
        print(f"Tokens: {len(generated.token_ids)}")
        print("-" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
