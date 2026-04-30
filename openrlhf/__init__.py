import importlib
import importlib.machinery
import os
import sys
import types


def _install_transformers_kernels_stub(reason: Exception) -> None:
    for name in list(sys.modules):
        if name == "kernels" or name.startswith("kernels."):
            sys.modules.pop(name, None)

    stub = types.ModuleType("kernels")
    stub.__spec__ = importlib.machinery.ModuleSpec("kernels", loader=None)
    stub.__dict__["__openrlhf_disabled__"] = True
    stub.__dict__["__openrlhf_disable_reason__"] = repr(reason)
    sys.modules["kernels"] = stub

    print(
        "[openrlhf] Disabling optional transformers 'kernels' integration "
        f"because importing 'kernels' failed: {reason!r}",
        file=sys.stderr,
    )


def _guard_optional_transformers_kernels() -> None:
    mode = os.environ.get("OPENRLHF_TRANSFORMERS_KERNELS", "auto").lower()
    if mode == "off":
        _install_transformers_kernels_stub(RuntimeError("disabled by OPENRLHF_TRANSFORMERS_KERNELS=off"))
        return
    if mode == "on":
        return

    try:
        importlib.import_module("kernels")
    except Exception as exc:
        _install_transformers_kernels_stub(exc)


_guard_optional_transformers_kernels()
