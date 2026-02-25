from .processor import get_processor, reward_normalization
from .utils import get_strategy, get_tokenizer


def __getattr__(name):
    # Lazy-load math_utils to avoid pulling in sympy/pylatexenc
    # when only lightweight submodules (e.g. tool_versions) are needed.
    if name in ("extract_boxed_answer", "grade_answer"):
        from .math_utils import extract_boxed_answer, grade_answer
        return extract_boxed_answer if name == "extract_boxed_answer" else grade_answer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "extract_boxed_answer",
    "get_processor",
    "grade_answer",
    "reward_normalization",
    "get_strategy",
    "get_tokenizer",
]
