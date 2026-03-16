from .processor import get_processor, reward_normalization
from .utils import get_strategy, get_tokenizer


#### Lazy-load math_utils to avoid pulling in sympy/pylatexenc for lightweight imports ####
def __getattr__(name):
    if name in ("extract_boxed_answer", "grade_answer"):
        from .math_utils import extract_boxed_answer, grade_answer
        return extract_boxed_answer if name == "extract_boxed_answer" else grade_answer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
#### end lazy-load math_utils ####


__all__ = [
    "extract_boxed_answer",
    "get_processor",
    "grade_answer",
    "reward_normalization",
    "get_strategy",
    "get_tokenizer",
]
