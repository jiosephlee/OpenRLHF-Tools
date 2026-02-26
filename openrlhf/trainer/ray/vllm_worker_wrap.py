class WorkerWrap:
    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name, backend="nccl", use_ray=False
    ):
        """Init torch process group for model weights update"""
        import torch
        from openrlhf.utils.distributed_util import stateless_init_process_group

        assert torch.distributed.is_initialized(), f"default torch process group must be initialized"
        assert group_name != "", f"group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset
        self._model_update_with_ray = use_ray
        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(world_size=world_size, rank=rank, backend=backend, group_name=group_name)
            self._model_update_group = group_name
        else:
            self._model_update_group = stateless_init_process_group(
                master_address,
                master_port,
                rank,
                world_size,
                self.device,
            )
        print(
            f"init_process_group: master_address={master_address}, master_port={master_port}, ",
            f"rank={rank}, world_size={world_size}, group_name={group_name}",
        )

    def _get_param_cache(self):
        if getattr(self, "_param_cache", None) is None:
            self._param_cache = dict(self.model_runner.model.named_parameters())
        return self._param_cache

    def _load_weight_into_model(self, name, weight):
        """Load a single weight tensor into the vLLM model, with fallback for buggy weight loaders.

        Works around a known vLLM bug where _load_weights_mxfp4() passes an unexpected
        `weight_name=` kwarg to default_weight_loader() (see ms-swift#7270).
        Falls back to direct param.data.copy_() with fuzzy name matching.
        """
        try:
            self.model_runner.model.load_weights(weights=[(name, weight)])
        except TypeError as e:
            if "unexpected keyword argument" not in str(e):
                raise
            if not getattr(self, "_fallback_warned", False):
                print(f"[WorkerWrap] load_weights TypeError workaround activated: {e}")
                self._fallback_warned = True
            state_dict = self._get_param_cache()
            
            # Use vLLM's own mapper to find the correct parameter name if possible
            mapped_name = name
            if hasattr(self.model_runner.model, "hf_to_vllm_mapper"):
                mapper = self.model_runner.model.hf_to_vllm_mapper
                if hasattr(mapper, "_mappings"):
                    # Basic lookup in the mappings dictionary if available
                    import re
                    for hf_pattern, vllm_pattern in mapper._mappings.items():
                        if isinstance(hf_pattern, re.Pattern):
                            match = hf_pattern.match(name)
                            if match:
                                # Apply replacement
                                if callable(vllm_pattern):
                                    res = vllm_pattern(match)
                                    if res:
                                        mapped_name = res
                                else:
                                    mapped_name = hf_pattern.sub(vllm_pattern, name)
                                break
                        elif hf_pattern == name:
                            mapped_name = vllm_pattern
                            break
            # Hardcoded fallbacks for MoE weights
            if mapped_name == name:
                if "gate_up_proj" in name:
                    mapped_name = name.replace("gate_up_proj", "w13_weight")
                elif "down_proj" in name:
                    mapped_name = name.replace("down_proj", "w2_weight")

            mapped_name = str(mapped_name)

            if mapped_name in state_dict:
                state_dict[mapped_name].data.copy_(weight)
            elif mapped_name + ".weight" in state_dict:
                state_dict[mapped_name + ".weight"].data.copy_(weight)
            else:
                matched = False
                for k, param in state_dict.items():
                    if k.endswith(mapped_name) or mapped_name.endswith(k) or k.replace(".weight", "") == mapped_name:
                        param.data.copy_(weight)
                        # Cache the successful fuzzy match so future lookups are O(1)
                        state_dict[mapped_name] = param
                        matched = True
                        break
                if not matched:
                    raise KeyError(f"Failed to find parameter {mapped_name} (original: {name}) for fallback. Available: {list(state_dict)[:5]}...")

    def update_weight(self, name, dtype, shape, empty_cache=False):
        import torch

        """Broadcast weight to all vllm workers from source rank 0 (actor model)"""
        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        if self._model_update_with_ray:
            import ray.util.collective as collective

            collective.broadcast(weight, 0, group_name=self._model_update_group)
        else:
            self._model_update_group.broadcast(weight, src=0, stream=torch.cuda.current_stream())

        self._load_weight_into_model(name, weight)

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles=None, empty_cache=False):
        import torch
        from openrlhf.trainer.ray.utils import get_physical_gpu_id

        if torch.distributed.get_rank() == 0:
            print(f"update weight: {name}, dtype: {dtype}, shape: {shape}")

        assert dtype == self.model_config.dtype, f"mismatch dtype: src {dtype}, dst {self.model_config.dtype}"

        handle = ipc_handles[get_physical_gpu_id()]
        device_id = self.device.index
        func, args = handle
        list_args = list(args)
        # the key is to change device id to the current device id
        # in case two processes have different CUDA_VISIBLE_DEVICES
        list_args[6] = device_id
        weight = func(*list_args)

        self._load_weight_into_model(name, weight)

        torch.cuda.synchronize()
