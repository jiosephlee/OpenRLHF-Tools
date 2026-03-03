import torch
from openrlhf.utils.nvfp4_quantize import _fake_quantize_nvfp4_chunk, _fake_quantize_nvfp4_chunk_v2, compute_nvfp4_global_scale, E2M1_VALUES

torch.manual_seed(42)
w = torch.randn((16, 2048, 4096), device="cuda", dtype=torch.bfloat16)
gs = compute_nvfp4_global_scale(w)
vals = E2M1_VALUES.to(w.device)
w_blocks = w.float().reshape(-1, 16)

out_ref = _fake_quantize_nvfp4_chunk(w_blocks, 16, gs, vals)
out_comp = _fake_quantize_nvfp4_chunk_v2(w_blocks, 16, gs, vals)

diff = (out_ref - out_comp).abs()
max_diff = diff.max().item()
print(f"Max diff: {max_diff}")

if max_diff > 0:
    idx = torch.where(diff == max_diff)
    print("Found mismatch at:", idx)
    i = idx[0][0].item()
    j = idx[1][0].item()
    print("w_flat:", w_blocks[i, j].item())
    
    # Let's trace it step by step for eager
    w_flat = w_blocks[i:i+1]
    
    # Eager trace
    vec_max = torch.max(torch.abs(w_flat), dim=-1, keepdim=True)[0].to(torch.float32)
    scale = vec_max / (6.0 * gs)
    scale = torch.clamp(scale, max=448.0, min=-448.0)
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    scaled_x = torch.where(scale == 0, torch.zeros_like(w_flat), w_flat / (scale * gs))
    clipped_x = torch.clamp(scaled_x, -6.0, 6.0)
    sign = torch.sign(clipped_x)
    abs_x = clipped_x.abs()
    ord_ = ((abs_x > 0.25).int() + (abs_x >= 0.75).int() + (abs_x > 1.25).int() + (abs_x >= 1.75).int() + (abs_x > 2.5).int() + (abs_x >= 3.5).int() + (abs_x > 5.0).int())
    dequantized = sign * vals[ord_]
    dequantized = dequantized * (scale * gs)
    print(f"Eager  -> scaled: {scaled_x[0, j].item()}, abs_x: {abs_x[0, j].item()}, ord: {ord_[0, j].item()}, res: {dequantized[0, j].item()}")

