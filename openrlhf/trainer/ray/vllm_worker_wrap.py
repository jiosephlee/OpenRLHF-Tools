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

    def _map_hf_name_to_vllm(self, name):
        """Map a HuggingFace parameter name to vLLM's internal name."""
        mapped_name = name
        if hasattr(self.model_runner.model, "hf_to_vllm_mapper"):
            mapper = self.model_runner.model.hf_to_vllm_mapper
            if hasattr(mapper, "_mappings"):
                import re
                for hf_pattern, vllm_pattern in mapper._mappings.items():
                    if isinstance(hf_pattern, re.Pattern):
                        match = hf_pattern.match(name)
                        if match:
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
            elif "down_proj" in name and "bias" not in name:
                mapped_name = name.replace("down_proj", "w2_weight")
        return str(mapped_name)

    def _find_param(self, mapped_name):
        """Look up a parameter by mapped name, with fuzzy matching fallback."""
        state_dict = self._get_param_cache()
        if mapped_name in state_dict:
            return state_dict[mapped_name]
        if mapped_name + ".weight" in state_dict:
            return state_dict[mapped_name + ".weight"]
        for k, param in state_dict.items():
            if k.endswith(mapped_name) or mapped_name.endswith(k) or k.replace(".weight", "") == mapped_name:
                state_dict[mapped_name] = param  # cache for O(1) next time
                return param
        return None

    def _is_mxfp4_expert_weight(self, mapped_name):
        """Check if this parameter is an MXFP4-quantized MoE expert weight (not scale or bias)."""
        return (
            ("w13_weight" in mapped_name or "w2_weight" in mapped_name or "gate_up_proj" in mapped_name or "down_proj" in mapped_name)
            and "_scale" not in mapped_name
            and "_bias" not in mapped_name
            and "bias" not in mapped_name
        )

    def _quantize_and_load_mxfp4(self, mapped_name, weight):
        """Quantize a bf16 expert weight to MXFP4 and write both packed weight + scale.

        The Actor's bf16 model is never modified; quantization happens on a copy
        during the broadcast to vLLM.
        """
        import torch
        from openrlhf.utils.mxfp4_quantize import quantize_to_mxfp4

        state_dict = self._get_param_cache()
        target_param = self._find_param(mapped_name)
        if target_param is None:
            raise KeyError(f"[MXFP4] Cannot find target param: {mapped_name}")

        # Determine corresponding scale parameter name
        # vLLM stores scales as e.g. "w13_weight_scale", "w2_weight_scale"
        scale_name = mapped_name + "_scale"
        scale_param = self._find_param(scale_name)
        if scale_param is None:
            raise KeyError(
                f"[MXFP4] Cannot find scale param for {mapped_name}. "
                f"Tried: {scale_name}. Available scale params: "
                f"{[k for k in state_dict if 'scale' in k][:5]}..."
            )

        # The checkpoint format stores weights transposed: [E, out_features, in_features]
        # but the Actor stores them as [E, in_features, out_features].
        # Transpose to match the checkpoint layout before quantizing.
        weight_t = weight.transpose(-1, -2).contiguous()

        target_shape = target_param.data.shape
        scale_shape = scale_param.data.shape

        # vLLM pads MoE weight dimensions for kernel alignment (e.g. to multiples of 512).
        # target_shape = [E, out_padded, in_packed_padded] where in_packed = in_features // 2
        out_padded = target_shape[1]
        in_padded = target_shape[2] * 2  # unpack: 2 FP4 values per uint8 byte
        out_actual, in_actual = weight_t.shape[1], weight_t.shape[2]
        
        if out_padded != out_actual or in_padded != in_actual:
            padded = torch.zeros(
                weight_t.shape[0], out_padded, in_padded,
                dtype=weight_t.dtype, device=weight_t.device,
            )
            padded[:, :out_actual, :in_actual] = weight_t
            weight_t = padded

        # Quantize each expert independently
        num_experts = weight_t.shape[0]
        packed_list = []
        scale_list = []
        for i in range(num_experts):
            packed, scales = quantize_to_mxfp4(weight_t[i], block_size=32)
            packed_list.append(packed)
            scale_list.append(scales)

        packed_weight = torch.stack(packed_list)
        packed_scales = torch.stack(scale_list)

        if not getattr(self, "_mxfp4_quantize_warned", False):
            pad_info = ""
            if out_padded != out_actual or in_padded != in_actual:
                pad_info = f" (zero-padded [{out_actual},{in_actual}]→[{out_padded},{in_padded}])"
            print(
                f"[MXFP4 Quantize] {mapped_name}: "
                f"bf16 {list(weight.shape)} → uint8 packed {list(packed_weight.shape)} "
                f"(target: {list(target_shape)}), "
                f"scales {list(packed_scales.shape)} (target: {list(scale_shape)}){pad_info}"
            )
            self._mxfp4_quantize_warned = True

        try:
            target_param.data.copy_(packed_weight.reshape(target_shape))
            scale_param.data.copy_(packed_scales.reshape(scale_shape))
        except RuntimeError as e:
            print(
                f"[MXFP4 Quantize] Shape mismatch for {mapped_name}: "
                f"packed={list(packed_weight.shape)}, target={list(target_shape)}, "
                f"scales={list(packed_scales.shape)}, scale_target={list(scale_shape)}. "
                f"Error: {e}"
            )
            raise

        # Mark that we need to re-run swizzling after all weights are synced
        if not hasattr(self, "_mxfp4_layers_dirty"):
            self._mxfp4_layers_dirty = set()
        # Extract layer index from name like "model.layers.0.mlp.experts.w13_weight"
        import re
        layer_match = re.search(r'layers\.(\d+)', mapped_name)
        if layer_match:
            self._mxfp4_layers_dirty.add(int(layer_match.group(1)))

    def reprocess_mxfp4_weights(self):
        """Re-run process_weights_after_loading() on MoE layers that received new MXFP4 weights.

        Must be called after all expert weights have been synced, to trigger
        vLLM's required swizzling/interleaving for the FlashInfer kernel.
        """
        dirty_layers = getattr(self, "_mxfp4_layers_dirty", set())
        if not dirty_layers:
            return

        model = self.model_runner.model
        reprocessed = 0
        for name, module in model.named_modules():
            # Look for FusedMoE layers (or similar) with a quant_method
            if hasattr(module, "quant_method") and hasattr(module.quant_method, "process_weights_after_loading"):
                # Check if this module's layer index is dirty
                import re
                layer_match = re.search(r'layers\.(\d+)', name)
                if layer_match and int(layer_match.group(1)) in dirty_layers:
                    try:
                        module.quant_method.process_weights_after_loading(module)
                        reprocessed += 1
                    except Exception as e:
                        print(f"[MXFP4] Warning: process_weights_after_loading failed for {name}: {e}")

        if reprocessed > 0:
            print(f"[MXFP4] Re-processed {reprocessed} MoE layers after weight sync")
        self._mxfp4_layers_dirty = set()

    def _load_weight_into_model(self, name, weight, mxfp4_quantize_on_the_fly=False):
        """Load a single weight tensor into the vLLM model.

        For MXFP4 expert weights: quantizes bf16→uint8 on the fly (Actor bf16 is unchanged).
        For everything else: direct copy with fallback for buggy weight loaders.
        """
        import torch

        mapped_name = self._map_hf_name_to_vllm(name)
        target_param = self._find_param(mapped_name)

        # Check if this is an MXFP4 expert weight that needs on-the-fly quantization
        # Only do this if the workflow explicitly enabled it via the CLI flag
        if mxfp4_quantize_on_the_fly and self._is_mxfp4_expert_weight(mapped_name) and weight.dtype in (torch.bfloat16, torch.float16):
            if target_param is not None and target_param.dtype == torch.uint8:
                self._quantize_and_load_mxfp4(mapped_name, weight)
                return

        # Standard path: try vLLM's load_weights first, fallback to direct copy
        try:
            self.model_runner.model.load_weights(weights=[(name, weight)])
        except TypeError as e:
            if "unexpected keyword argument" not in str(e):
                raise
            if not getattr(self, "_fallback_warned", False):
                print(f"[WorkerWrap] load_weights TypeError workaround activated: {e}")
                self._fallback_warned = True

            if target_param is not None:
                target_param.data.copy_(weight)
            else:
                raise KeyError(f"Failed to find parameter {mapped_name} (original: {name}) for fallback. Available: {list(self._get_param_cache())[:5]}...")

    def update_weight(self, name, dtype, shape, empty_cache=False, mxfp4_quantize_on_the_fly=False):
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

        self._load_weight_into_model(name, weight, mxfp4_quantize_on_the_fly)

        del weight
        # TODO: should we empty cache if all weights have updated?
        # if empty_cache:
        #     torch.cuda.empty_cache()

    def update_weight_cuda_ipc(self, name, dtype, shape, ipc_handles=None, empty_cache=False, mxfp4_quantize_on_the_fly=False):
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

        self._load_weight_into_model(name, weight, mxfp4_quantize_on_the_fly)

        torch.cuda.synchronize()
