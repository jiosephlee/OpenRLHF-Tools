# Attention Backend Benchmarking

Self-contained scripts for benchmarking HuggingFace model speeds with different attention backends.

## Scripts

| Script | What it measures |
|---|---|
| `benchmark_forward.py` | Inference-only forward pass (no gradients) |
| `benchmark_backward.py` | Training forward + backward pass (with gradients) |

## Backends Tested

- **eager** — vanilla multi-head attention (reference baseline)
- **sdpa** — `torch.nn.functional.scaled_dot_product_attention`
- **flash_attention_2** — Flash Attention 2 (requires `flash-attn`)
- **flex_attention** — PyTorch FlexAttention (requires `torch >= 2.5`)

## Quick Start

```bash
# Forward pass benchmark
python benchmarking/benchmark_forward.py \
    --model meta-llama/Llama-3.1-8B \
    --seq_lengths 512 1024 2048 4096

# Backward pass benchmark
python benchmarking/benchmark_backward.py \
    --model meta-llama/Llama-3.1-8B \
    --seq_lengths 512 1024 2048

# Specific backends only
python benchmarking/benchmark_forward.py \
    --model Qwen/Qwen3-8B \
    --backends sdpa flash_attention_2 \
    --seq_lengths 1024 2048 4096

# With gradient checkpointing (backward only)
python benchmarking/benchmark_backward.py \
    --model meta-llama/Llama-3.1-8B \
    --gradient_checkpointing \
    --seq_lengths 2048 4096 8192

# Save CSV results
python benchmarking/benchmark_forward.py \
    --model meta-llama/Llama-3.1-8B \
    --csv forward_results.csv
```

## Options (both scripts)

| Flag | Default | Description |
|---|---|---|
| `--model` | (required) | HF model name or local path |
| `--backends` | all four | Space-separated list of backends |
| `--seq_lengths` | 512 1024 2048 4096 | Sequence lengths to sweep |
| `--batch_size` | 1 | Batch size |
| `--dtype` | bf16 | fp32, fp16, or bf16 |
| `--warmup` | 3 | Warmup iterations |
| `--iters` | 10 | Benchmark iterations |
| `--csv` | None | Save results to CSV |

**Backward-only:**

| Flag | Default | Description |
|---|---|---|
| `--gradient_checkpointing` | off | Trade ~30% speed for memory savings |

## Output

Both scripts print:
1. **Per-backend results** as they run (latency, memory, throughput)
2. **Summary table** comparing all backends
3. **Speedup vs eager** relative comparison
4. **Memory savings** (backward script only)
