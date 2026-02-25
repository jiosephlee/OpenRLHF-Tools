# OpenRLHF-Tools Troubleshooting Guide

## Ray Actor `Socket closed` / `ActorUnavailableError` during Optimizer Initialization
**Symptoms:**
- During training initialization, specifically right after "Before initializing optimizer states" is logged (or right before).
- Ray throws an `ActorUnavailableError` saying `RpcError: RPC Error message: Socket closed`.

**Root Cause:**
- This typically points to the Linux OOM-killer silently terminating the Ray actor process because it tried to allocate more memory than the OS (or SLURM) allowed.
- In particular, **DeepSpeed's `--adam_offload`** pins the optimizer states (momentum and variance) to CPU RAM to save GPU VRAM.
- For a 20B parameter model (e.g., `gpt-oss`), the Adam optimizer FP32 states take roughly `20B * 8 bytes = 160GB` of CPU memory.
- In **colocated mode** (or a multi-GPU actor setup like `8aXv`), DeepSpeed ZeRO-2/3 shards this 160GB optimizer state across all actor processes. For 8 GPUs, each process only attempts to pin `160GB / 8 = 20GB` of CPU RAM, which easily fits within typical SLURM limits (e.g., `--mem-per-gpu=128G`).
- However, in **distributed `1a1v` mode** (1 GPU dedicated to the actor), the single actor process must hold the *entire* 160GB of optimizer states in its own CPU RAM space. If the SLURM job was allocated 128GB per GPU, 160GB > 128GB, and the process is killed instantly.

**Solutions:**
1. **Remove `--adam_offload`:** If you use a dedicated actor GPU in distributed mode (e.g. `1a1v`), it doesn't share VRAM with vLLM, so the optimizer states can likely fit directly in VRAM. Removing the flag bypasses CPU offloading completely and avoids the CPU OOM.
2. **Increase Actor GPUs:** Shard the actor across multiple GPUs (e.g., `ACTOR_GPUS=2`), cutting the per-process CPU memory requirement in half (to 80GB), which safely fits inside a 128GB allocation.
3. **Increase CPU RAM Allocation:** If using `--adam_offload` with a 1-GPU actor is absolutely necessary, request significantly more memory from SLURM (e.g., `--mem=256G` or `--mem-per-gpu=256G`).

## Raylet Killed (SIGKILL) During vLLM Engine Initialization
**Symptoms:**
- Raylet terminates unexpectedly shortly after vLLM engines begin loading (you'll see model weights loaded successfully, then `Dynamo bytecode transform time: ~15s`, then the Raylet dies).
- The error says "Possible reasons include: (1) SIGKILL by the user or system OOM killer" with only Ray state-dump lines — no Python traceback.
- You may also see `ray.exceptions.ActorUnavailableError: The actor is temporarily unavailable: RpcError: RPC Error message: Socket closed` — this means the actor's process was killed out from under Ray by the OS OOM killer.
- Typically happens when **multiple vLLM engines initialize concurrently** in colocated mode.

**Root Cause:**
- **Newer vLLM versions** (especially builds from git main) aggressively pre-allocate GPU memory for CUDAGraph capture. The default config captures **84 different batch sizes** (1 through 1024) using `FULL_AND_PIECEWISE` CUDAGraph mode.
- Each capture allocates temporary GPU memory on top of model weights + KV cache (`gpu_memory_utilization`), easily exceeding available VRAM when two engines init simultaneously.
- The worker process hits a CUDA OOM, crashes, and the Raylet is killed — often with no Python traceback, just a silent SIGKILL.

**Solutions:**
1. **`--reduce_cuda_graph`** (preferred): Reduces CUDAGraph capture sizes from 84 to `[1, 2, 4, 8, 16, 32]` via `CompilationConfig`. Keeps torch.compile benefits with much lower init-time memory.
2. **`--enforce_eager`:** Disables torch.compile and CUDAGraph entirely. ~10-15% generation slowdown but zero OOM risk during init.
3. **Lower `--vllm_gpu_memory_utilization`:** Reduces KV cache pre-allocation (e.g., 0.7 → 0.5) to leave more headroom for CUDAGraph capture. May reduce throughput.

**Note:** Environment variables like `VLLM_CUDAGRAPH_CAPTURE_SIZES` are **not recognized** by all vLLM builds. Use the `--reduce_cuda_graph` CLI flag instead, which passes a `CompilationConfig` directly through the Python API.

## FlashInfer JIT Cache Errors (`libcudart.so` / `Sparsity` / Ninja Build Failed)
**Symptoms:**
- `RuntimeError: Failed to load dynamic shared library .../fp4_quantization_100.so libcudart.so.13: cannot open shared object file`
- `namespace "batchedGemm::trtllm::gen" has no member "Sparsity"` (100 compilation errors)
- Ninja build commands referencing the **wrong conda env** (e.g., `open_rlhf_intern` paths when you activated `openrlhf`)

**Root Cause:**
- FlashInfer JIT-compiles CUDA kernels on first use and caches them under `~/.cache/flashinfer/`. The cached `.so` files and `build.ninja` scripts bake in the CUDA version and conda env paths from the session that originally compiled them.
- Switching CUDA versions (e.g., 12.8 → 13.1) or conda environments makes the cache stale — shared libraries link against the wrong `libcudart`, and build files reference non-existent include paths.
- Cubin header mismatches (like missing `Sparsity`) indicate the downloaded cubins are incompatible with the installed FlashInfer version.

**Solution:**
```bash
rm -rf ~/.cache/flashinfer/
```
Then re-run. FlashInfer will re-download cubins and JIT-recompile kernels against the currently loaded CUDA and active conda env.