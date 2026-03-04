import torch
import triton
import triton.language as tl
from openrlhf.utils.nvfp4_quantize import quantize_to_nvfp4, _fake_quantize_nvfp4_chunk, _fake_quantize_nvfp4_triton, compute_nvfp4_global_scale

@triton.jit
def _triton_fused_nvfp4_kernel(
    w_ptr, global_scale_ptr, out_ptr, N,
    BLOCK_SIZE: tl.constexpr,
    BLOCKS_PER_PROGRAM: tl.constexpr,
):
    pid = tl.program_id(0)
    
    global_scale = tl.load(global_scale_ptr).to(tl.float32)
    
    for i in range(BLOCKS_PER_PROGRAM):
        block_idx = pid * BLOCKS_PER_PROGRAM + i
        base_offs = block_idx * BLOCK_SIZE
        offs = base_offs + tl.arange(0, BLOCK_SIZE)
        mask = offs < N

        if base_offs < N:
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            
            abs_w = tl.abs(w)
            vec_max = tl.max(abs_w)
            
            # log2 based? No, it's just / 6.0 / global_scale
            scale = vec_max / (6.0 * global_scale)
            scale = tl.minimum(tl.maximum(scale, -448.0), 448.0)
            
            # cast to E4M3 and back
            scale = scale.to(tl.float8e4nv).to(tl.float32)
            
            combined_scale = scale * global_scale

            scaled_x = tl.where(combined_scale == 0.0, 0.0, tl.div_rn(w, combined_scale))
            clipped_x = tl.minimum(tl.maximum(scaled_x, -6.0), 6.0)
            abs_x = tl.abs(clipped_x)

            ord_ = ((abs_x > 0.25).to(tl.int32) + (abs_x >= 0.75).to(tl.int32) +
                    (abs_x > 1.25).to(tl.int32) + (abs_x >= 1.75).to(tl.int32) +
                    (abs_x > 2.5).to(tl.int32) + (abs_x >= 3.5).to(tl.int32) +
                    (abs_x > 5.0).to(tl.int32))

            m_bit = (ord_ & 1).to(tl.float32)
            e_bits = (ord_ >> 1).to(tl.float32)
            normal_val = (1.0 + m_bit * 0.5) * tl.math.exp2(e_bits - 1.0)
            subnormal_val = ord_.to(tl.float32) * 0.5
            q_val = tl.where(ord_ < 2, subnormal_val, normal_val)

            sign = tl.where(clipped_x >= 0, 1.0, -1.0)
            result = sign * q_val * combined_scale

            tl.store(out_ptr + offs, result.to(tl.bfloat16), mask=mask)

def _fake_quantize_nvfp4_triton_fused(weight: torch.Tensor, block_size: int, global_scale: torch.Tensor) -> torch.Tensor:
    original_shape = weight.shape
    N = weight.numel()
    num_blocks = triton.cdiv(N, block_size)
    assert N % block_size == 0

    w_flat = weight.reshape(-1)
    out = torch.empty(N, dtype=torch.bfloat16, device=weight.device)
    
    BLOCKS_PER_PROGRAM = 64
    num_programs = triton.cdiv(num_blocks, BLOCKS_PER_PROGRAM)
    grid = (num_programs,)

    gs_tensor = global_scale.view(1)

    _triton_fused_nvfp4_kernel[grid](
        w_flat, gs_tensor, out, N,
        BLOCK_SIZE=block_size,
        BLOCKS_PER_PROGRAM=BLOCKS_PER_PROGRAM,
    )
    return out.reshape(original_shape)


def test_fused():
    torch.manual_seed(42)
    w = torch.randn((16, 2048, 4096), device="cuda", dtype=torch.bfloat16)
    
    gs = compute_nvfp4_global_scale(w)
    out_ref = _fake_quantize_nvfp4_triton(w, 16, gs)
    out_fused = _fake_quantize_nvfp4_triton_fused(w, 16, gs)
    
    diff = (out_ref - out_fused).abs().max().item()
    print(f"Max diff: {diff}")
    if diff == 0:
        print("Success! Fused kernel matches.")
    else:
        print("Mismatch.")

if __name__ == "__main__":
    test_fused()
