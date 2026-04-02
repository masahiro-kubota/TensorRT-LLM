_OPTIONAL_IMPORT_ERROR = None

try:
    from .llm import LLM
except Exception as exc:  # pragma: no cover - probe-only compatibility path
    _OPTIONAL_IMPORT_ERROR = exc
    LLM = None

__all__ = ["LLM"]
