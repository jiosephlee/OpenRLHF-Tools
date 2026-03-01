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

## SIGKILL / OOM During Checkpoint Loading with Adam Offload
**Symptoms:**
- SLURM job receives a SIGKILL (or Ray actor crashes) specifically while trying to load a checkpoint (`load_checkpoint`), even when your node has plenty of physical RAM (e.g., 768GB).
- This happens closely following initialization.

**Root Cause:**
- When `--adam_offload` is used, DeepSpeed allocates pinned CPU memory for the optimizer states. For a 20B model on 1 GPU, this is roughly 160GB of pinned RAM.
- When loading a checkpoint, `torch.load` reads the optimizer state dictionary from disk into *normal pageable CPU RAM* before it is copied into DeepSpeed's pre-allocated pinned buffers.
- This creates a massive memory spike: `160GB (pinned) + 160GB (load buffer) = 320GB` peak memory requirement for a brief moment.
- Even if your node has 768GB RAM, **SLURM cgroups strictly enforce your requested limit**. If your `#SBATCH` memory request is `--mem=256G`, the OOM-killer will terminate your process the second `torch.load` pushes it over the 256GB limit.

**Solutions:**
1. **Dramatically Increase System RAM:** Bump your SLURM memory request significantly to cover this doubling spike (e.g., to `--mem=450G` or `--mem=512G`).
2. **Increase Actor GPUs:** Shard the actor across multiple GPUs (e.g., `ACTOR_GPUS=2`). This slices both the pinned memory and the `torch.load` buffer size in half per process (80GB + 80GB = 160GB peak per process), keeping it safely within tighter SLURM allocations.
3. **Switch to 8-bit Adam (`--adam_8bit`):** This keeps optimizer states on the GPU instead of offloading to CPU. Note: While it eliminates the CPU RAM spike, it is still being evaluated for training overhead/speed and increases VRAM requirements.


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

## vLLM Compilation Errors (`Bytes object is corrupted` / `AssertionError` in `standalone_compile.py`)
**Symptoms:**
- `RuntimeError: Bytes object is corrupted, checksum does not match. Expected: b'h\xda/\x19', Got: b'\x07\x0e\xccL'`
- `AssertionError: CacheInfo(artifacts=defaultdict(<class 'list'>, {'autotune': [...], 'aot_autograd': []}))` in `torch._inductor/standalone_compile.py` during engine startup.

**Root Cause:**
- PyTorch inductor cache corruption or stale compiled artifacts in the `.cache/torch/inductor` directory. 
- Previous interrupted runs can leave corrupted or structurally incompatible artifacts (such as empty AOT autograd artifact sets for some models) that cause subsequent vLLM engine initializations to crash when loading or saving compiled graphs.

**Solution:**
Clear the `torch.inductor` and vLLM compile caches before running (this has already been added to the SLURM training scripts):
```bash
rm -rf ~/.cache/torch/inductor/ /tmp/torchinductor_${USER}/ ~/.cache/vllm/torch_compile_cache/
```

## Training Jobs Running 3-4x Slower Than Expected
**Symptoms:**
- The evaluation `tqdm` bar or generation rollouts take proportionally much longer (e.g., 16 minutes instead of 5 minutes) consistently across an entire run.
- W&B metrics for "Network Traffic" and "Disk I/O" climb linearly but at very low actual bandwidths (e.g., ~1MB/s).

**Root Cause:**
- **CPU Core Starvation by SLURM allocations:** Even if you request `--cpus-per-gpu=12`, SLURM's `cgroups` might artificially restrict your job to a fraction of the requested physical cores on densely packed nodes. `vLLM` and `DeepSpeed` are highly CPU-bound; if they are forced to share 4 or 8 physical cores, the constant Python thread context-switching leaves the GPUs entirely idle, causing massive artificial bottlenecks.
- To verify this, run `nvidia-smi topo -m` and look at the `CPU Affinity` column for your assigned GPUs. If the number of logical threads assigned is significantly lower than your requested `--cpus-per-gpu * gpus`, you are being starved.

**Solutions:**
1. **Force Exclusive Cores:** Add `#SBATCH --exclusive` or `#SBATCH --core-spec=0` (if allowed by your cluster admin) to guarantee dedicated physical cores.
2. **Increase CPU Requests:** Bump `--cpus-per-gpu` higher (e.g. to 12 or 16) to force SLURM to find a node with enough free continuous cores.