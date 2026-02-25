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
