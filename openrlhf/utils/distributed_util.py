def torch_dist_barrier_and_cuda_sync():
    """Synchronize distributed training and CUDA operations.
    This function ensures that:
    1. All distributed processes reach this point (barrier)
    2. All CUDA operations are completed (synchronize)
    """
    import torch

    torch.distributed.barrier()
    torch.cuda.synchronize()


def stateless_init_process_group(master_address, master_port, rank, world_size, device):
    """
    vLLM provides `StatelessProcessGroup` to create a process group
    without considering the global process group in torch.distributed.
    It is recommended to create `StatelessProcessGroup`, and then initialize
    the data-plane communication (NCCL) between external (train processes)
    and vLLM workers.
    """
    import os
    import torch
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    #### NCCL debug diagnostics ####
    # Force NCCL debug for this call so it propagates to subprocesses
    os.environ.setdefault("NCCL_DEBUG", "INFO")

    # Diagnostic: log exactly what each process sees
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "<not set>")
    current_dev = torch.cuda.current_device() if torch.cuda.is_initialized() else "<not initialized>"
    print(
        f"[stateless_init_process_group] rank={rank}, world_size={world_size}, "
        f"device={device}, CUDA_VISIBLE_DEVICES={cuda_visible}, "
        f"torch.cuda.current_device()={current_dev}, "
        f"master={master_address}:{master_port}, pid={os.getpid()}",
        flush=True,
    )
    #### end NCCL debug diagnostics ####

    pg = StatelessProcessGroup.create(host=master_address, port=master_port, rank=rank, world_size=world_size)
    print(f"[stateless_init_process_group] rank={rank}: StatelessProcessGroup created OK, calling PyNcclCommunicator...", flush=True)
    pynccl = PyNcclCommunicator(pg, device=device)
    print(f"[stateless_init_process_group] rank={rank}: PyNcclCommunicator created OK", flush=True)
    return pynccl
