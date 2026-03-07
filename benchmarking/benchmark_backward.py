#!/usr/bin/env python3
"""
Benchmark Backward Pass Speed — Attention Backends & Kernels

Self-contained script to benchmark the backward pass (gradient computation)
of any HuggingFace causal-LM with different attention implementations.

Backends tested:
  - eager        : vanilla multi-head attention (reference, slowest)
  - sdpa         : torch.nn.functional.scaled_dot_product_attention
  - flash_attention_2 : Flash Attention 2 (requires flash-attn package)
  - flex_attention    : PyTorch FlexAttention (torch >= 2.5)

Usage:
  python benchmark_backward.py --model meta-llama/Llama-3.1-8B \
      --seq_lengths 512 1024 2048 --batch_size 1 --dtype bf16

  # Subset of backends
  python benchmark_backward.py --model Qwen/Qwen3-8B \
      --backends eager sdpa flash_attention_2 --seq_lengths 1024

  # Gradient checkpointing (saves memory, costs ~30% speed)
  python benchmark_backward.py --model meta-llama/Llama-3.1-8B \
      --seq_lengths 2048 4096 --gradient_checkpointing

  # Save CSV
  python benchmark_backward.py --model meta-llama/Llama-3.1-8B \
      --seq_lengths 512 1024 2048 --csv results_bwd.csv
"""

import argparse
import csv
import gc
import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────

DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

ALL_BACKENDS = ["eager", "sdpa", "flash_attention_2", "flex_attention"]


def check_backend_available(backend: str) -> Tuple[bool, str]:
    """Return (available, reason) for a given backend."""
    if backend == "eager":
        return True, ""
    if backend == "sdpa":
        return True, ""
    if backend == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
            return True, ""
        except ImportError:
            return False, "flash-attn package not installed"
    if backend == "flex_attention":
        if hasattr(torch.nn.functional, "flex_attention"):
            return True, ""
        if torch.__version__ >= "2.5":
            return True, ""
        return False, f"requires torch >= 2.5 (have {torch.__version__})"
    return False, f"unknown backend: {backend}"


def gpu_mem_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def reset_peak_memory():
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_max_memory_allocated()


@contextmanager
def cuda_timer():
    """Context manager that yields a dict with 'elapsed_ms' after exit."""
    result = {}
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    yield result
    end.record()
    torch.cuda.synchronize()
    result["elapsed_ms"] = start.elapsed_time(end)


def make_dummy_input(
    batch_size: int, seq_len: int, vocab_size: int, device: str
) -> Dict[str, torch.Tensor]:
    """Generate random input_ids + attention_mask + labels for loss computation."""
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    # Labels = shifted input_ids (standard causal LM training)
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


# ──────────────────────────────────────────────────────────────────────
# Core benchmark
# ──────────────────────────────────────────────────────────────────────

def load_model(
    model_name: str,
    backend: str,
    dtype: torch.dtype,
    gradient_checkpointing: bool = False,
    device: str = "cuda",
):
    """Load a model with a specific attention implementation, in training mode."""
    kwargs = dict(
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    if backend in ("eager", "sdpa", "flash_attention_2", "flex_attention"):
        kwargs["attn_implementation"] = backend

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    model.train()  # Training mode for backward pass
    return model


def benchmark_backward_pass(
    model,
    inputs: Dict[str, torch.Tensor],
    warmup_iters: int = 3,
    bench_iters: int = 10,
) -> Dict[str, float]:
    """
    Returns dict with:
      fwd_mean_ms, fwd_std_ms   — forward pass times
      bwd_mean_ms, bwd_std_ms   — backward pass times
      total_mean_ms, total_std_ms — forward + backward
      peak_mem_mb                — peak GPU memory
    """
    # Warmup (full fwd + bwd)
    for _ in range(warmup_iters):
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()

    # Benchmark
    fwd_times = []
    bwd_times = []
    reset_peak_memory()

    for _ in range(bench_iters):
        # Forward
        with cuda_timer() as fwd_t:
            outputs = model(**inputs)
            loss = outputs.loss

        # Backward
        with cuda_timer() as bwd_t:
            loss.backward()

        fwd_times.append(fwd_t["elapsed_ms"])
        bwd_times.append(bwd_t["elapsed_ms"])

        model.zero_grad(set_to_none=True)

    peak_mem = gpu_mem_mb()

    def stats(vals):
        mean = sum(vals) / len(vals)
        std = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5
        return mean, std

    fwd_mean, fwd_std = stats(fwd_times)
    bwd_mean, bwd_std = stats(bwd_times)
    total_times = [f + b for f, b in zip(fwd_times, bwd_times)]
    total_mean, total_std = stats(total_times)

    return {
        "fwd_mean_ms": fwd_mean,
        "fwd_std_ms": fwd_std,
        "bwd_mean_ms": bwd_mean,
        "bwd_std_ms": bwd_std,
        "total_mean_ms": total_mean,
        "total_std_ms": total_std,
        "peak_mem_mb": peak_mem,
    }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def run_benchmarks(args):
    dtype = DTYPE_MAP[args.dtype]
    device = "cuda"

    backends = args.backends if args.backends else ALL_BACKENDS

    available_backends = []
    for b in backends:
        ok, reason = check_backend_available(b)
        if ok:
            available_backends.append(b)
        else:
            print(f"⚠️  Skipping {b}: {reason}")

    if not available_backends:
        print("❌ No backends available. Exiting.")
        sys.exit(1)

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    vocab_size = config.vocab_size

    print(f"\n{'=' * 90}")
    print(f"  BACKWARD PASS BENCHMARK")
    print(f"{'=' * 90}")
    print(f"  Model              : {args.model}")
    print(f"  dtype              : {args.dtype}")
    print(f"  Batch size         : {args.batch_size}")
    print(f"  Seq lengths        : {args.seq_lengths}")
    print(f"  Backends           : {available_backends}")
    print(f"  Grad checkpointing : {args.gradient_checkpointing}")
    print(f"  Warmup iters       : {args.warmup}")
    print(f"  Bench iters        : {args.iters}")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_total = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
    print(f"  GPU                : {gpu_name} ({gpu_mem_total:.1f} GB)")
    print(f"{'=' * 90}\n")

    results: List[dict] = []

    for backend in available_backends:
        print(f"\n{'─' * 70}")
        print(f"  Loading model with attn_implementation = {backend}")
        if args.gradient_checkpointing:
            print(f"  Gradient checkpointing: ENABLED")
        print(f"{'─' * 70}")

        gc.collect()
        torch.cuda.empty_cache()

        try:
            model = load_model(
                args.model, backend, dtype,
                gradient_checkpointing=args.gradient_checkpointing,
                device=device,
            )
        except Exception as e:
            print(f"  ❌ Failed to load with {backend}: {e}")
            continue

        # Identify attention class
        attn_class = "unknown"
        for module in model.modules():
            cls_name = type(module).__name__
            if "attention" in cls_name.lower() and "layer" not in cls_name.lower():
                attn_class = cls_name
                break

        print(f"  Attention class: {attn_class}")

        for seq_len in args.seq_lengths:
            inputs = make_dummy_input(args.batch_size, seq_len, vocab_size, device)

            try:
                metrics = benchmark_backward_pass(
                    model, inputs,
                    warmup_iters=args.warmup,
                    bench_iters=args.iters,
                )
                throughput = (args.batch_size * seq_len) / (metrics["total_mean_ms"] / 1000)

                row = {
                    "backend": backend,
                    "attn_class": attn_class,
                    "seq_len": seq_len,
                    "batch_size": args.batch_size,
                    "grad_ckpt": args.gradient_checkpointing,
                    **metrics,
                    "throughput_tok_s": throughput,
                }
                results.append(row)

                print(
                    f"  seq={seq_len:>5}  |  "
                    f"fwd {metrics['fwd_mean_ms']:>7.2f}ms  "
                    f"bwd {metrics['bwd_mean_ms']:>7.2f}ms  "
                    f"total {metrics['total_mean_ms']:>7.2f}ms  |  "
                    f"{metrics['peak_mem_mb']:>8.1f} MB  |  "
                    f"{throughput:>10,.0f} tok/s"
                )

            except torch.cuda.OutOfMemoryError:
                print(f"  seq={seq_len:>5}  |  ❌ OOM")
                results.append({
                    "backend": backend,
                    "attn_class": attn_class,
                    "seq_len": seq_len,
                    "batch_size": args.batch_size,
                    "grad_ckpt": args.gradient_checkpointing,
                    "fwd_mean_ms": float("nan"),
                    "fwd_std_ms": float("nan"),
                    "bwd_mean_ms": float("nan"),
                    "bwd_std_ms": float("nan"),
                    "total_mean_ms": float("nan"),
                    "total_std_ms": float("nan"),
                    "peak_mem_mb": float("nan"),
                    "throughput_tok_s": float("nan"),
                })
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"  seq={seq_len:>5}  |  ❌ Error: {e}")
                torch.cuda.empty_cache()

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary table ─────────────────────────────────────────────
    print(f"\n\n{'=' * 120}")
    print(f"  SUMMARY — Backward Pass (fwd + bwd)")
    print(f"{'=' * 120}")
    header = (
        f"{'Backend':<20} {'Attn Class':<30} {'SeqLen':>7} "
        f"{'Fwd (ms)':>10} {'Bwd (ms)':>10} {'Total (ms)':>11} "
        f"{'Bwd/Fwd':>8} {'Mem (MB)':>10} {'Tok/s':>12}"
    )
    print(header)
    print("─" * 120)
    for r in results:
        if r["total_mean_ms"] != r["total_mean_ms"]:  # NaN
            print(
                f"{r['backend']:<20} {r['attn_class']:<30} {r['seq_len']:>7}       OOM"
            )
        else:
            bwd_fwd_ratio = r["bwd_mean_ms"] / r["fwd_mean_ms"] if r["fwd_mean_ms"] > 0 else 0
            print(
                f"{r['backend']:<20} {r['attn_class']:<30} {r['seq_len']:>7} "
                f"{r['fwd_mean_ms']:>10.2f} {r['bwd_mean_ms']:>10.2f} "
                f"{r['total_mean_ms']:>11.2f} {bwd_fwd_ratio:>8.2f}x "
                f"{r['peak_mem_mb']:>10.1f} {r['throughput_tok_s']:>12,.0f}"
            )

    # ── Relative comparison ───────────────────────────────────────
    eager_times = {
        r["seq_len"]: r["total_mean_ms"]
        for r in results if r["backend"] == "eager"
    }
    if eager_times:
        print(f"\n{'─' * 60}")
        print(f"  Speedup vs Eager (total = fwd + bwd)")
        print(f"{'─' * 60}")
        for r in results:
            if r["backend"] == "eager" or r["total_mean_ms"] != r["total_mean_ms"]:
                continue
            base = eager_times.get(r["seq_len"])
            if base and base == base:
                speedup = base / r["total_mean_ms"]
                print(
                    f"  {r['backend']:<20} seq={r['seq_len']:>5}  →  {speedup:.2f}x"
                )

    # ── Memory comparison ─────────────────────────────────────────
    eager_mem = {
        r["seq_len"]: r["peak_mem_mb"]
        for r in results if r["backend"] == "eager"
    }
    if eager_mem:
        print(f"\n{'─' * 60}")
        print(f"  Memory Savings vs Eager")
        print(f"{'─' * 60}")
        for r in results:
            if r["backend"] == "eager" or r["peak_mem_mb"] != r["peak_mem_mb"]:
                continue
            base = eager_mem.get(r["seq_len"])
            if base and base == base:
                saving_pct = (1 - r["peak_mem_mb"] / base) * 100
                print(
                    f"  {r['backend']:<20} seq={r['seq_len']:>5}  →  "
                    f"{saving_pct:+.1f}% ({r['peak_mem_mb']:.0f} vs {base:.0f} MB)"
                )

    # ── CSV output ────────────────────────────────────────────────
    if args.csv:
        fieldnames = [
            "backend", "attn_class", "seq_len", "batch_size", "grad_ckpt",
            "fwd_mean_ms", "fwd_std_ms", "bwd_mean_ms", "bwd_std_ms",
            "total_mean_ms", "total_std_ms", "peak_mem_mb", "throughput_tok_s",
        ]
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\n📄 Results saved to {args.csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark backward pass with different attention backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--backends", nargs="+", default=None,
        choices=ALL_BACKENDS,
        help=f"Attention backends to test (default: all of {ALL_BACKENDS})",
    )
    parser.add_argument(
        "--seq_lengths", nargs="+", type=int,
        default=[512, 1024, 2048],
        help="Sequence lengths to benchmark (default: 512 1024 2048)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--dtype", type=str, default="bf16", choices=list(DTYPE_MAP.keys()),
        help="Model dtype (default: bf16)",
    )
    parser.add_argument(
        "--gradient_checkpointing", action="store_true",
        help="Enable gradient checkpointing (trades ~30%% speed for memory)",
    )
    parser.add_argument(
        "--warmup", type=int, default=3,
        help="Number of warmup iterations (default: 3)",
    )
    parser.add_argument(
        "--iters", type=int, default=10,
        help="Number of benchmark iterations (default: 10)",
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to save results as CSV",
    )
    args = parser.parse_args()
    run_benchmarks(args)


if __name__ == "__main__":
    main()
