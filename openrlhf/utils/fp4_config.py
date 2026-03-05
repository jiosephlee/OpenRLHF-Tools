"""FP4 quantization configuration dataclass.

Centralises all FP4-related flags that were previously scattered across
train_ppo_ray.py validation, actor.py, and ppo_actor.py.

Usage::

    # In train_ppo_ray.py validation block:
    args.fp4_config = FP4Config.from_args(args)
    args.fp4_config.validate()

    # In actor.py / ppo_actor.py:
    fp4_config = getattr(strategy.args, "fp4_config", None)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FP4Config:
    """Consolidated FP4 quantization settings.

    Attributes:
        sync_format: Format used for on-the-fly weight sync to vLLM.
            None | "mxfp4" | "nvfp4"
        qat_enabled: Whether FP4 fake-quantization QAT is active.
        dequantize_base: True when the base model was loaded from a
            pre-quantized checkpoint and dequantized to bf16 for training
            (i.e. --mxfp4_dequantize or --nvfp4_dequantize_base_model).
    """

    sync_format: Optional[str]
    qat_enabled: bool
    dequantize_base: bool

    @classmethod
    def from_args(cls, args) -> "FP4Config":
        sync_format = getattr(args, "vllm_sync_fp4", None)
        qat_enabled = getattr(args, "qat", None) == "fp4_fake_quantize"
        dequantize_base = getattr(args, "mxfp4_dequantize", False) or bool(
            getattr(args, "nvfp4_dequantize_base_model", None)
        )
        return cls(sync_format=sync_format, qat_enabled=qat_enabled, dequantize_base=dequantize_base)

    def validate(self) -> None:
        if self.qat_enabled and not self.sync_format:
            raise ValueError(
                "--qat fp4_fake_quantize requires --vllm_sync_fp4 to be set (mxfp4 or nvfp4) "
                "so the FP4 format for fake quantization is known."
            )
        if self.qat_enabled and self.sync_format == "mxfp4" and not self.dequantize_base:
            raise ValueError(
                "--qat fp4_fake_quantize with mxfp4 requires --mxfp4_dequantize. The model must be "
                "loaded with Mxfp4Config(dequantize=True) so expert weights are in bf16."
            )
