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

    def debug_weight_snapshot(self, label=""):
        """Print a fingerprint of key model weights for debugging weight sync.

        Call before and after weight sync to detect:
        - Weights that didn't change (sync missed them)
        - Weights that became NaN/Inf (corruption)
        - Weights that are all zeros (failed to load)
        """
        import torch

        model = self.model_runner.model
        print(f"\n[DEBUG WeightSnapshot] {label}")
        print(f"  model class: {model.__class__.__name__}")

        # Check a few representative weights from different categories
        snapshot_names = []
        for name, param in model.named_parameters():
            # Sample: first layer's layernorm, first expert weight, embedding, lm_head
            if any(k in name for k in [
                "layers.0.input_layernorm.weight",
                "layers.0.mlp.experts.w13_weight",
                "layers.0.mlp.experts.w2_weight",
                "layers.0.mlp.experts.w13_weight_scale",
                "layers.0.mlp.experts.w13_bias",
                "layers.0.attn.qkv_proj.weight",
                "embedding.weight",
                "lm_head.weight",
            ]):
                snapshot_names.append((name, param))

        for name, param in snapshot_names:
            data = param.data
            if data.is_meta:
                print(f"  {name}: META DEVICE (not materialized!)")
            elif data.numel() == 0:
                print(f"  {name}: EMPTY shape={list(data.shape)}")
            else:
                flat = data.detach().cpu().float().flatten()
                print(
                    f"  {name}: shape={list(data.shape)} dtype={data.dtype} "
                    f"device={data.device} "
                    f"mean={flat.mean().item():.6f} std={flat.std().item():.6f} "
                    f"absmax={flat.abs().max().item():.6f} "
                    f"nan={flat.isnan().any().item()} inf={flat.isinf().any().item()} "
                    f"allzero={flat.eq(0).all().item()} "
                    f"hash={flat[:8].tolist()}"  # first 8 values as fingerprint
                )
                del flat
        print()

    def _maybe_quantize_for_vllm(self, name, weight):
        """If MXFP4 is active and this is a MoE expert weight, quantize bf16 → uint8.

        Yields (name, tensor) pairs suitable for load_weights():
          - For expert weights: yields the packed uint8 weight AND its E8M0 scales
          - For everything else: yields the original (name, weight) unchanged

        Names are kept in HF convention — vLLM's hf_to_vllm_mapper handles
        remapping (e.g. gate_up_proj → w13_weight, gate_up_proj_scales → w13_weight_scale).

        The Actor stores expert weights as [E, in_features, out_features] in bf16.
        The checkpoint expects [E, out_features, in_features] packed as uint8.

        Note: gate_up_proj data is already interleaved [gate_0, up_0, gate_1, up_1, ...]
        from the checkpoint. The Actor preserves this layout during training, so no
        interleaving is needed here. vLLM's process_weights_after_loading applies
        swap_every_two_rows to swap w1↔w3 for trtllm-gen's swiglu convention.
        """
        import torch

        # Identify MoE expert weight names (HF convention)
        is_gate_up = "gate_up_proj" in name
        is_down = "down_proj" in name
        is_expert_weight = (
            (is_gate_up or is_down)
            and "_scale" not in name
            and "bias" not in name
        )

        if not is_expert_weight or weight.dtype not in (torch.bfloat16, torch.float16):
            yield name, weight
            return

        from openrlhf.utils.mxfp4_quantize import quantize_to_mxfp4

        # Transpose: Actor [E, in, out] → checkpoint [E, out, in]
        weight_t = weight.transpose(-1, -2).contiguous()

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

        # Use HF naming convention — vLLM's hf_to_vllm_mapper handles the rest:
        #   gate_up_proj → w13_weight, gate_up_proj_scales → w13_weight_scale
        #   down_proj → w2_weight, down_proj_scales → w2_weight_scale
        scale_name = name.replace("gate_up_proj", "gate_up_proj_scales") if is_gate_up else name.replace("down_proj", "down_proj_scales")

        if not getattr(self, "_mxfp4_quantize_logged", False):
            print(
                f"[MXFP4 Quantize] {name}: bf16 {list(weight.shape)} → "
                f"uint8 packed {list(packed_weight.shape)}, "
                f"scales {list(packed_scales.shape)}"
            )
            self._mxfp4_quantize_logged = True

        yield name, packed_weight
        yield scale_name, packed_scales

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

        if mxfp4_quantize_on_the_fly:
            for w_name, w_tensor in self._maybe_quantize_for_vllm(name, weight):
                self.model_runner.model.load_weights(weights=[(w_name, w_tensor)])
        else:
            self.model_runner.model.load_weights(weights=[(name, weight)])

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

        if mxfp4_quantize_on_the_fly:
            for w_name, w_tensor in self._maybe_quantize_for_vllm(name, weight):
                self.model_runner.model.load_weights(weights=[(w_name, w_tensor)])
        else:
            self.model_runner.model.load_weights(weights=[(name, weight)])

        torch.cuda.synchronize()

    def initialize_weight_reload(self):
        """Prepare model for layerwise weight reloading.

        Must be called before the weight sync loop. Saves kernel-format tensors,
        restores parameters to model format (meta device), and wraps weight
        loaders so that per-layer process_weights_after_loading runs automatically
        as weights arrive.
        """
        import torch
        from vllm.model_executor.model_loader.reload import initialize_layerwise_reload

        self.debug_weight_snapshot("BEFORE weight sync (pre-initialize_layerwise_reload)")
        with torch.device(self.device):
            initialize_layerwise_reload(self.model_runner.model)
        print("[WorkerWrap] initialize_weight_reload: layerwise reload initialized")

    def post_weight_sync(self):
        """Finalize layerwise reload after all weights are synced.

        Unwraps layerwise weight loaders, processes any remaining layers
        (Attention/MLA), and restores kernel tensors.
        """
        import torch
        from vllm.model_executor.model_loader.reload import finalize_layerwise_reload

        with torch.device(self.device):
            finalize_layerwise_reload(self.model_runner.model, self.model_config)
        torch.cuda.synchronize()
        self.debug_weight_snapshot("AFTER weight sync (post-finalize_layerwise_reload)")
        print("[WorkerWrap] post_weight_sync: finalize_layerwise_reload complete")
