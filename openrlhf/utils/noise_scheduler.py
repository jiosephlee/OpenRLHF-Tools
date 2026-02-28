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
    from vllm.model_executor.layers.layernorm import RMSNorm

    for name, module in model.named_modules():
        if isinstance(module, RMSNorm):
            weight_tensor = module.weight
            sigma_id, sigma = get_sigma_by_step(step, total_step, sigma_trend)
            # print("Current AQN step:", step, "Total steps:", total_step, "Sigma id:", sigma_id, "Sigma:", sigma)
            if sigma == 0:
                return
            noise = torch.normal(mean=0, std=sigma, size=weight_tensor.shape, dtype=torch.float32).to(
                weight_tensor.device
            )
            noise = noise.to(weight_tensor.dtype)
            with torch.no_grad():
                module.weight.add_(noise)
