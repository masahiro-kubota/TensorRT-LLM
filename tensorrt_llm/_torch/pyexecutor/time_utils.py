from tensorrt_llm.bindings import steady_clock_now


def get_steady_clock_now_in_seconds() -> float:
    return steady_clock_now().total_seconds()
