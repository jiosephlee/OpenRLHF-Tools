import torch


def get_sigma_by_step(step, total_steps, sigma_trend):
    step = min(step, total_steps)

    num_intervals = len(sigma_trend) + 1
    steps_per_interval = total_steps / num_intervals

    interval_id = int(step // steps_per_interval)

    if interval_id == 0:
        return interval_id, 0

    sigma_id = interval_id - 1
    sigma_id = min(sigma_id, len(sigma_trend) - 1)

    sigma = sigma_trend[sigma_id]
    return sigma_id, sigma


def generate_gaussian_noise(model, step, total_step, sigma_trend):
    """Inject Gaussian noise into RMSNorm/LayerNorm weights to simulate quantization (QeRL).

    Works with both HuggingFace transformers and vLLM models by matching
    any module whose name contains 'norm' and has a `weight` attribute.
    """
    # Unwrap Actor wrapper if needed — get the underlying nn.Module
    underlying = getattr(model, "model", model)
    # Unwrap DeepSpeed engine if present
    underlying = getattr(underlying, "module", underlying)

    for name, module in underlying.named_modules():
        if "norm" not in name.lower():
            continue
        if not hasattr(module, "weight") or module.weight is None:
            continue

        weight_tensor = module.weight
        sigma_id, sigma = get_sigma_by_step(step, total_step, sigma_trend)
        if sigma == 0:
            return
        noise = torch.normal(mean=0, std=sigma, size=weight_tensor.shape, dtype=torch.float32).to(weight_tensor.device)
        noise = noise.to(weight_tensor.dtype)
        with torch.no_grad():
            module.weight.add_(noise)
