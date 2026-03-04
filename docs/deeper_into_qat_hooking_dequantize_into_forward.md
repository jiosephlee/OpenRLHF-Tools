# Deeper Into QAT: Hooking Fake‑Dequantization Into Forward

How `nn.utils.parametrize`, the pre‑forward cache hook, and gradient
checkpointing interact during a single training step.

---

## 1. The Setup (model init, before training)

`register_mxfp4_qat_parametrization()` calls:

```python
parametrize.register_parametrization(
    module, "gate_up_proj",
    _Mxfp4FakeQuant(block_size, transpose=True),
)
```

This is a **one‑time structural transformation** that does three things:

1. **Moves the original tensor** — the raw `nn.Parameter` that was
   `module.gate_up_proj` gets relocated to
   `module.parametrizations.gate_up_proj.original`.
   This is the actual learned parameter the optimizer sees and updates.

2. **Replaces the attribute with a Python property** — `module.gate_up_proj`
   is no longer a plain tensor attribute.  It becomes a `@property` descriptor
   whose getter calls
   `_Mxfp4FakeQuant.forward(module.parametrizations.gate_up_proj.original)`
   **every time it is accessed**.

3. **Stores the parametrization module** — the `_Mxfp4FakeQuant` instance is
   added to a `ParametrizationList` at
   `module.parametrizations.gate_up_proj[0]`.

After registration the module looks conceptually like:

```python
class GptOssExperts(nn.Module):
    # Before: gate_up_proj was nn.Parameter([E, in, out])
    # After:
    #   self.parametrizations.gate_up_proj.original  ← nn.Parameter (optimizer target)
    #   self.parametrizations.gate_up_proj[0]        ← _Mxfp4FakeQuant instance

    @property      # installed by register_parametrization
    def gate_up_proj(self):
        return self.parametrizations.gate_up_proj[0](
            self.parametrizations.gate_up_proj.original
        )
```

---

## 2. Where Parametrize Fires in the Timeline

The key insight: **parametrize does not use hooks**.  It uses Python's
descriptor protocol — a plain property access.

```
Training step N:

  model.forward() called
  ├── pre-forward hook fires (cache management — separate mechanism)
  │
  ├── layer 0 runs
  │     └── GptOssExperts.forward() executes:
  │           out = x @ self.gate_up_proj.transpose(-1, -2)
  │                       ^^^^^^^^^^^^^^^^^^^^
  │                       This is a PROPERTY ACCESS, not a tensor read.
  │
  │           Python descriptor protocol kicks in:
  │           ├── module.__class__.gate_up_proj.__get__(module) called
  │           │     ├── retrieves module.parametrizations.gate_up_proj.original
  │           │     │   (the real bf16 nn.Parameter, the one the optimizer updates)
  │           │     │
  │           │     └── calls _Mxfp4FakeQuant.forward(original_weight)
  │           │           ├── fake_quantize_mxfp4(weight)
  │           │           │     ├── Triton kernel: bf16 → E8M0 scale → E2M1 snap
  │           │           │     │                  → dequant → bf16
  │           │           │     └── returns dequantized tensor
  │           │           └── return weight + (dequantized - weight).detach()
  │           │                              ↑ STE trick: forward sees quantized,
  │           │                                backward gradient flows to original
  │           │
  │           └── result is the fake-quantized tensor, used in matmul
```

---

## 3. Why This Design Matters

| Aspect | What `parametrize` gives you |
|--------|------------------------------|
| **Transparency** | Layer code (`GptOssExperts.forward`) doesn't know fake quant exists — it just accesses `self.gate_up_proj` normally |
| **Optimizer correctness** | The optimizer sees `module.parametrizations.gate_up_proj.original` as the parameter. Gradients flow through the STE to this original bf16 weight |
| **Composability** | Multiple parametrizations can stack (though here there's only one) |
| **Gradient‑checkpoint safe** | Since it's a property, it fires on every access — including recomputation. No hook to miss |

---

## 4. The Interaction With the Pre‑Forward Cache

The pre-forward hook registered on `model` (not individual layers) manages an
optional `_dq_cache` on each `_Mxfp4FakeQuant` instance. When the cache is
present the property‑access flow becomes:

```
Property access: self.gate_up_proj
  → _Mxfp4FakeQuant.forward(original_weight)
    → check self._dq_cache
    → if cached: return cached value  (no Triton kernel)
    → if not:    run fake_quantize_mxfp4(), store result, return it
```

Without caching, every property access (forward **plus** each gradient
checkpoint recompute) re-runs the Triton kernel.  Correct but wasteful.

With caching, the pre‑forward hook seeds the cache once per step, and every
subsequent property access (forward and recompute) is a cheap cache hit.

---

## 5. Full Step Timeline (with cache)

```
Training step N:

  model.forward() called
  ├── pre-forward hook fires
  │     ├── clear _dq_cache on all 64 fq instances  (frees step N-1's tensors)
  │     └── launch 64 Triton kernels on 8 streams   (async, GPU starts immediately)
  │
  ├── layer 0 runs
  │     └── self.gate_up_proj accessed
  │           → Python property fires
  │           → _Mxfp4FakeQuant.forward(original_weight)
  │               → wait_stream (GPU sync if kernel not done yet)
  │               → dq = self._dq_cache   ← cache hit, not cleared
  │               → returns weight + (dq - weight).detach()
  │
  ├── layers 1–31 run similarly
  │     all cache hits, stream already done → wait is no-op
  │
  loss.backward() called
  ├── checkpointing recomputes layer 5
  │     └── layer_5.forward() called  ← NOT model.forward(), hook does NOT fire
  │           └── self.gate_up_proj accessed
  │                 → Python property fires again
  │                 → _Mxfp4FakeQuant.forward(original_weight)
  │                     → _dq_cache is still set (we never cleared it)
  │                     → wait_stream is a no-op (kernel finished long ago)
  │                     → same result as the original forward pass ✓
  │
  └── all other recomputes similarly use the cache

Training step N+1:

  model.forward() called
  └── pre-forward hook fires → clears caches → launches new kernels
```

---

## 6. Why the Cache Survives the Backward Pass

Three properties guarantee this:

1. **The pre‑forward hook only runs on `model.forward()`** — once per step.
2. **`_Mxfp4FakeQuant.forward()` never clears `_dq_cache`** — it only reads.
3. **Gradient checkpointing recomputes call `layer.forward()`**, not
   `model.forward()`, so the hook does not fire during recompute.

The second `wait_stream` during recompute is a no‑op since the CUDA stream
completed during the forward pass.  The recompute gets the exact same
fake‑quantized tensor, so the STE gradient is consistent with the original
forward.

---

## 7. Summary: Parametrize as the Bridge

```
register_parametrization() [one-time setup]
  ↓
Transforms module.gate_up_proj from nn.Parameter → @property
  ↓
Every access to module.gate_up_proj now calls:
  _Mxfp4FakeQuant.forward(original_weight)
  ↓
  Returns: weight + (dequant - weight).detach()  [STE]
  ↓
Layer code uses the result as if it were a normal tensor
  ↓
Autograd sees the STE → gradients flow to original weight
  ↓
Optimizer updates original weight normally
```

The parametrize mechanism is the **bridge** between "the layer thinks it has a
normal weight" and "the weight is actually being fake‑quantized on every
access."  The cache / stream / hook infrastructure then optimizes **when** that
fake‑quantization computation actually happens relative to when the result is
consumed.
